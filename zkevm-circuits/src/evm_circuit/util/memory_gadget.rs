use crate::{
    evm_circuit::{
        param::{N_BYTES_GAS, N_BYTES_MEMORY_SIZE},
        util::{
            constraint_builder::ConstraintBuilder,
            from_bytes,
            math_gadget::IsZeroGadget,
            math_gadget::{ConstantDivisionGadget, MaxGadget},
            sum, MemoryAddress,
        },
    },
    util::Expr,
};
use array_init::array_init;
use bus_mapping::evm::GasCost;
use halo2::plonk::Error;
use halo2::{arithmetic::FieldExt, circuit::Region, plonk::Expression};

/// Decodes the usable part of an address stored in a Word
pub(crate) mod address_low {
    use crate::evm_circuit::{
        param::N_BYTES_MEMORY_ADDRESS,
        util::{from_bytes, Word},
    };
    use halo2::{arithmetic::FieldExt, plonk::Expression};

    pub(crate) fn expr<F: FieldExt>(address: &Word<F>) -> Expression<F> {
        from_bytes::expr(&address.cells[..N_BYTES_MEMORY_ADDRESS])
    }

    pub(crate) fn value<F: FieldExt>(address: [u8; 32]) -> u64 {
        from_bytes::value::<F>(&address[..N_BYTES_MEMORY_ADDRESS])
            .get_lower_128() as u64
    }
}

/// The sum of bytes of the address that are unused for most calculations on the
/// address
pub(crate) mod address_high {
    use crate::evm_circuit::{
        param::N_BYTES_MEMORY_ADDRESS,
        util::{sum, Word},
    };
    use halo2::{arithmetic::FieldExt, plonk::Expression};

    pub(crate) fn expr<F: FieldExt>(address: &Word<F>) -> Expression<F> {
        sum::expr(&address.cells[N_BYTES_MEMORY_ADDRESS..])
    }

    pub(crate) fn value<F: FieldExt>(address: [u8; 32]) -> F {
        sum::value::<F>(&address[N_BYTES_MEMORY_ADDRESS..])
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MemoryAddressGadget<F> {
    length: MemoryAddress<F>,
    length_is_zero: IsZeroGadget<F>,
    offset_bytes: MemoryAddress<F>,
    address: Expression<F>,
}

impl<F: FieldExt> MemoryAddressGadget<F> {
    pub(crate) fn construct(
        cb: &mut ConstraintBuilder<F>,
        offset: Expression<F>,
        length: MemoryAddress<F>,
    ) -> Self {
        let length_is_zero =
            IsZeroGadget::construct(cb, sum::expr(&length.cells));
        let offset_bytes = cb.query_rlc();

        cb.condition(1.expr() - length_is_zero.expr(), |cb| {
            cb.require_equal(
                "Offset decomposition",
                offset_bytes.expr(),
                offset,
            );
        });

        let address = (1.expr() - length_is_zero.expr())
            * (from_bytes::expr(&offset_bytes.cells)
                + from_bytes::expr(&length.cells));

        Self {
            length,
            length_is_zero,
            offset_bytes,
            address,
        }
    }

    pub(crate) fn address(&self) -> Expression<F> {
        self.address.expr()
    }
}

/// Calculates the memory size required for a memory access at the specified
/// address. `memory_size = ceil(address/32) = floor((address + 31) / 32)`
#[derive(Clone, Debug)]
pub(crate) struct MemorySizeGadget<F> {
    memory_size: ConstantDivisionGadget<F, N_BYTES_MEMORY_SIZE>,
}

impl<F: FieldExt> MemorySizeGadget<F> {
    pub(crate) fn construct(
        cb: &mut ConstraintBuilder<F>,
        address: Expression<F>,
    ) -> Self {
        let memory_size =
            ConstantDivisionGadget::construct(cb, address + 31.expr(), 32);

        Self { memory_size }
    }

    pub(crate) fn expr(&self) -> Expression<F> {
        self.memory_size.quotient()
    }

    pub(crate) fn assign(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        address: u64,
    ) -> Result<u64, Error> {
        let (quotient, _) =
            self.memory_size
                .assign(region, offset, (address as u128) + 31)?;
        Ok(quotient as u64)
    }
}

/// Returns (new memory size, memory gas cost) for a memory access.
/// If the memory needs to be expanded this will result in an extra gas cost.
/// This gas cost is the difference between the next and current memory costs:
/// `memory_cost = Gmem * memory_size + floor(memory_size * memory_size / 512)`
#[derive(Clone, Debug)]
pub(crate) struct MemoryExpansionGadget<
    F,
    const N: usize,
    const N_BYTES_MEMORY_SIZE: usize,
> {
    memory_sizes: [MemorySizeGadget<F>; N],
    max_memory_sizes: [MaxGadget<F, N_BYTES_MEMORY_SIZE>; N],
    curr_quad_memory_cost: ConstantDivisionGadget<F, N_BYTES_GAS>,
    next_quad_memory_cost: ConstantDivisionGadget<F, N_BYTES_GAS>,
    next_memory_size: Expression<F>,
    gas_cost: Expression<F>,
}

impl<F: FieldExt, const N: usize, const N_BYTES_MEMORY_SIZE: usize>
    MemoryExpansionGadget<F, N, N_BYTES_MEMORY_SIZE>
{
    /// Input requirements:
    /// - `curr_memory_size < 256**MAX_MEMORY_SIZE_IN_BYTES`
    /// - `address < 32 * 256**MAX_MEMORY_SIZE_IN_BYTES`
    /// Output ranges:
    /// - `next_memory_size < 256**MAX_MEMORY_SIZE_IN_BYTES`
    /// - `gas_cost <= GAS_MEM*256**MAX_MEMORY_SIZE_IN_BYTES +
    ///   256**MAX_QUAD_COST_IN_BYTES`
    pub(crate) fn construct(
        cb: &mut ConstraintBuilder<F>,
        curr_memory_size: Expression<F>,
        addresses: [Expression<F>; N],
    ) -> Self {
        // Calculate the memory size of the memory access
        // `address_memory_size < 256**MAX_MEMORY_SIZE_IN_BYTES`
        let memory_sizes =
            addresses.map(|address| MemorySizeGadget::construct(cb, address));

        // The memory size needs to be updated if this memory access
        // requires expanding the memory.
        // `next_memory_size < 256**MAX_MEMORY_SIZE_IN_BYTES`
        let mut next_memory_size = curr_memory_size.expr();
        let max_memory_sizes = array_init(|idx| {
            let max_memory_size = MaxGadget::construct(
                cb,
                next_memory_size.expr(),
                memory_sizes[idx].expr(),
            );
            next_memory_size = max_memory_size.expr();
            max_memory_size
        });

        // Calculate the quad memory cost for the current and next memory size.
        // These quad costs will also be range limited to `<
        // 256**MAX_QUAD_COST_IN_BYTES`.
        let curr_quad_memory_cost = ConstantDivisionGadget::construct(
            cb,
            curr_memory_size.expr() * curr_memory_size.expr(),
            GasCost::MEMORY_EXPANSION_QUAD_DENOMINATOR.as_u64(),
        );
        let next_quad_memory_cost = ConstantDivisionGadget::construct(
            cb,
            next_memory_size.expr() * next_memory_size.expr(),
            GasCost::MEMORY_EXPANSION_QUAD_DENOMINATOR.as_u64(),
        );

        // Calculate the gas cost for the memory expansion.
        // This gas cost is the difference between the next and current memory
        // costs. `gas_cost <=
        // GAS_MEM*256**MAX_MEMORY_SIZE_IN_BYTES + 256**MAX_QUAD_COST_IN_BYTES`
        let gas_cost = GasCost::MEMORY_EXPANSION_LINEAR_COEFF.expr()
            * (next_memory_size.expr() - curr_memory_size)
            + (next_quad_memory_cost.quotient()
                - curr_quad_memory_cost.quotient());

        Self {
            memory_sizes,
            max_memory_sizes,
            curr_quad_memory_cost,
            next_quad_memory_cost,
            next_memory_size,
            gas_cost,
        }
    }

    pub(crate) fn next_memory_size(&self) -> Expression<F> {
        // Return the new memory size
        self.next_memory_size.expr()
    }

    pub(crate) fn gas_cost(&self) -> Expression<F> {
        // Return the gas cost
        self.gas_cost.expr()
    }

    pub(crate) fn assign(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        curr_memory_size: u32,
        addresses: [u64; N],
    ) -> Result<(u64, u64), Error> {
        // Calculate the active memory size
        let memory_sizes = self
            .memory_sizes
            .iter()
            .zip(addresses.iter())
            .map(|(memory_size, address)| {
                memory_size.assign(region, offset, *address)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Calculate the next memory size
        let mut next_memory_size = curr_memory_size as u64;
        for (max_memory_sizes, memory_size) in
            self.max_memory_sizes.iter().zip(memory_sizes.iter())
        {
            next_memory_size = max_memory_sizes
                .assign(
                    region,
                    offset,
                    F::from(next_memory_size as u64),
                    F::from(*memory_size),
                )?
                .get_lower_128() as u64;
        }

        // Calculate the quad gas cost for the memory size
        let (curr_quad_memory_cost, _) = self.curr_quad_memory_cost.assign(
            region,
            offset,
            (curr_memory_size as u128) * (curr_memory_size as u128),
        )?;
        let (next_quad_memory_cost, _) = self.next_quad_memory_cost.assign(
            region,
            offset,
            (next_memory_size as u128) * (next_memory_size as u128),
        )?;

        // Calculate the gas cost for the expansian
        let memory_cost = GasCost::MEMORY_EXPANSION_LINEAR_COEFF.as_u64()
            * (next_memory_size - curr_memory_size as u64)
            + (next_quad_memory_cost - curr_quad_memory_cost) as u64;

        // Return the new memory size and the memory expansion gas cost
        Ok((next_memory_size, memory_cost))
    }
}
