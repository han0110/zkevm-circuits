use crate::{
    evm_circuit::{
        execution::ExecutionGadget,
        param::{
            EMPTY_HASH, N_BYTES_ACCOUNT_ADDRESS, N_BYTES_GAS,
            N_BYTES_MEMORY_SIZE,
        },
        step::ExecutionState,
        table::{AccountFieldTag, CallContextFieldTag, FixedTableTag, Lookup},
        util::{
            common_gadget::TransferGadget,
            constraint_builder::{
                ConstraintBuilder, StepStateTransition,
                Transition::{Delta, To},
            },
            from_bytes,
            math_gadget::{
                ConstantDivisionGadget, IsEqualGadget, IsZeroGadget,
                MinMaxGadget,
            },
            memory_gadget::{MemoryAddressGadget, MemoryExpansionGadget},
            select, sum, Cell, Word,
        },
        witness::{Block, Call, ExecStep, Transaction},
    },
    util::Expr,
};
use bus_mapping::{
    eth_types::{ToLittleEndian, ToScalar},
    evm::GasCost,
};
use halo2::{arithmetic::FieldExt, circuit::Region, plonk::Error};

#[derive(Clone, Debug)]
pub(crate) struct CallGadget<F> {
    opcode: Cell<F>,
    tx_id: Cell<F>,
    rw_counter_end_of_reversion: Cell<F>,
    is_persistent: Cell<F>,
    caller_address: Cell<F>,
    is_static: Cell<F>,
    depth: Cell<F>,
    gas: Word<F>,
    callee_address: Word<F>,
    value: Word<F>,
    is_success: Cell<F>,
    gas_is_u64: IsZeroGadget<F>,
    is_warm_access: Cell<F>,
    is_warm_access_prev: Cell<F>,
    callee_rw_counter_end_of_reversion: Cell<F>,
    callee_is_persistent: Cell<F>,
    value_is_zero: IsZeroGadget<F>,
    cd_address: MemoryAddressGadget<F>,
    rd_address: MemoryAddressGadget<F>,
    memory_expansion: MemoryExpansionGadget<F, 2, N_BYTES_MEMORY_SIZE>,
    transfer: TransferGadget<F>,
    callee_nonce: Cell<F>,
    callee_code_hash: Cell<F>,
    callee_nonce_is_zero: IsZeroGadget<F>,
    callee_balance_is_zero: IsZeroGadget<F>,
    callee_code_hash_is_empty: IsEqualGadget<F>,
    one_64th_gas: ConstantDivisionGadget<F, N_BYTES_GAS>,
    capped_callee_gas_left: MinMaxGadget<F, N_BYTES_GAS>,
}

impl<F: FieldExt> ExecutionGadget<F> for CallGadget<F> {
    const NAME: &'static str = "CALL";

    const EXECUTION_STATE: ExecutionState = ExecutionState::CALL;

    fn configure(cb: &mut ConstraintBuilder<F>) -> Self {
        let opcode = cb.query_cell();
        cb.opcode_lookup(opcode.expr(), 1.expr());
        cb.add_lookup(Lookup::Fixed {
            tag: FixedTableTag::ResponsibleOpcode.expr(),
            values: [
                cb.execution_state().as_u64().expr(),
                opcode.expr(),
                0.expr(),
            ],
        });

        let gas_word = cb.query_word();
        let callee_address_word = cb.query_word();
        let value = cb.query_word();
        let cd_offset = cb.query_cell();
        let cd_length = cb.query_rlc();
        let rd_offset = cb.query_cell();
        let rd_length = cb.query_rlc();
        let is_success = cb.query_bool();

        // Use rw_counter of the step which triggers next call as its call_id.
        let callee_call_id = cb.curr.state.rw_counter.clone();

        let [tx_id, rw_counter_end_of_reversion, is_persistent, caller_address, is_static, depth] =
            [
                CallContextFieldTag::TxId,
                CallContextFieldTag::RwCounterEndOfReversion,
                CallContextFieldTag::IsPersistent,
                CallContextFieldTag::CallerAddress,
                CallContextFieldTag::IsStatic,
                CallContextFieldTag::Depth,
            ]
            .map(|field_tag| cb.call_context(None, field_tag));

        cb.range_lookup(depth.expr() - 1.expr(), 1024);

        // Lookup values from stack
        cb.stack_pop(gas_word.expr());
        cb.stack_pop(callee_address_word.expr());
        cb.stack_pop(value.expr());
        cb.stack_pop(cd_offset.expr());
        cb.stack_pop(cd_length.expr());
        cb.stack_pop(rd_offset.expr());
        cb.stack_pop(rd_length.expr());
        cb.stack_push(is_success.expr());

        // Recomposition of random linear combination to integer
        let callee_address = from_bytes::expr(
            &callee_address_word.cells[..N_BYTES_ACCOUNT_ADDRESS],
        );
        let gas = from_bytes::expr(&gas_word.cells[..N_BYTES_GAS]);
        let gas_is_u64 = IsZeroGadget::construct(
            cb,
            sum::expr(&gas_word.cells[N_BYTES_GAS..]),
        );
        let cd_address =
            MemoryAddressGadget::construct(cb, cd_offset, cd_length);
        let rd_address =
            MemoryAddressGadget::construct(cb, rd_offset, rd_length);
        let memory_expansion = MemoryExpansionGadget::construct(
            cb,
            cb.curr.state.memory_size.expr(),
            [cd_address.address(), rd_address.address()],
        );

        // Add callee to access list
        let is_warm_access = cb.query_bool();
        let is_warm_access_prev = cb.query_bool();
        let is_cold_access = cb.account_access_list_write_with_reversion(
            tx_id.expr(),
            callee_address.clone(),
            is_warm_access.expr(),
            is_warm_access_prev.expr(),
            is_persistent.expr(),
            rw_counter_end_of_reversion.expr(),
        );

        // Propagate rw_counter_end_of_reversion and is_persistent
        let [callee_rw_counter_end_of_reversion, callee_is_persistent] = [
            CallContextFieldTag::RwCounterEndOfReversion,
            CallContextFieldTag::IsPersistent,
        ]
        .map(|field_tag| {
            cb.call_context(Some(callee_call_id.expr()), field_tag)
        });
        cb.require_equal(
            "callee_is_persistent == is_persistent ⋅ is_success",
            callee_is_persistent.expr(),
            is_persistent.expr() * is_success.expr(),
        );
        cb.condition(is_success.expr() * (1.expr() - is_persistent.expr()), |cb| {
            cb.require_equal(
                "callee_rw_counter_end_of_reversion == rw_counter_end_of_reversion - (state_write_counter + 1)",
                callee_rw_counter_end_of_reversion.expr(),
                rw_counter_end_of_reversion.expr()
                    - (cb.curr.state.state_write_counter.expr() + 1.expr()),
            );
        });

        // Verify transfer
        let value_is_zero =
            IsZeroGadget::construct(cb, sum::expr(&value.cells));
        let has_value = 1.expr() - value_is_zero.expr();
        cb.condition(has_value.clone(), |cb| {
            cb.require_zero(
                "CALL with value must not be in static call stack",
                is_static.expr(),
            );
        });
        let transfer = TransferGadget::construct(
            cb,
            caller_address.expr(),
            callee_address.clone(),
            value.clone(),
            is_persistent.expr(),
            rw_counter_end_of_reversion.expr(),
        );

        // Verify gas cost
        let [callee_nonce, callee_code_hash] =
            [AccountFieldTag::Nonce, AccountFieldTag::CodeHash].map(
                |field_tag| {
                    let value = cb.query_cell();
                    cb.account_read(
                        callee_address.clone(),
                        field_tag,
                        value.expr(),
                    );
                    value
                },
            );
        let callee_nonce_is_zero = IsZeroGadget::construct(
            cb,
            sum::expr(&transfer.receiver_balance_prev().cells),
        );
        let callee_balance_is_zero =
            IsZeroGadget::construct(cb, callee_nonce.expr());
        let callee_code_hash_is_empty = IsEqualGadget::construct(
            cb,
            callee_code_hash.expr(),
            Word::random_linear_combine_expr(
                EMPTY_HASH.map(|byte| byte.expr()),
                cb.power_of_randomness(),
            ),
        );
        let is_account_empty = callee_nonce_is_zero.expr()
            * callee_balance_is_zero.expr()
            * callee_code_hash_is_empty.expr();
        let gas_cost = GasCost::WARM_ACCESS.expr()
            + is_cold_access * GasCost::EXTRA_COLD_ACCESS_ACCOUNT.expr()
            + is_account_empty * GasCost::CALL_EMPTY_ACCOUNT.expr()
            + has_value.clone() * GasCost::CALL_WITH_VALUE.expr()
            + memory_expansion.gas_cost();

        // Apply EIP 150
        let gas_available = cb.curr.state.gas_left.expr() - gas_cost.clone();
        let one_64th_gas =
            ConstantDivisionGadget::construct(cb, gas_available.clone(), 64);
        let all_but_one_64th_gas = gas_available - one_64th_gas.quotient();
        let capped_callee_gas_left =
            MinMaxGadget::construct(cb, gas, all_but_one_64th_gas.clone());
        let callee_gas_left = select::expr(
            gas_is_u64.expr(),
            capped_callee_gas_left.min(),
            all_but_one_64th_gas,
        );

        // TODO: Handle precompiled

        // Save caller's call state
        for (field_tag, value) in [
            (CallContextFieldTag::IsRoot, cb.curr.state.is_root.expr()),
            (
                CallContextFieldTag::IsCreate,
                cb.curr.state.is_create.expr(),
            ),
            (
                CallContextFieldTag::CodeSource,
                cb.curr.state.code_source.expr(),
            ),
            (
                CallContextFieldTag::ProgramCounter,
                cb.curr.state.program_counter.expr() + 1.expr(),
            ),
            (
                CallContextFieldTag::StackPointer,
                cb.curr.state.stack_pointer.expr() + 6.expr(),
            ),
            (
                CallContextFieldTag::GasLeft,
                cb.curr.state.gas_left.expr()
                    - gas_cost
                    - callee_gas_left.clone(),
            ),
            (
                CallContextFieldTag::MemorySize,
                memory_expansion.next_memory_size(),
            ),
            (
                CallContextFieldTag::StateWriteCounter,
                cb.curr.state.state_write_counter.expr() + 1.expr(),
            ),
        ] {
            cb.call_context_lookup(true.expr(), None, field_tag, value);
        }

        // Setup next call's context.
        for (field_tag, value) in [
            (
                CallContextFieldTag::CallerCallId,
                cb.curr.state.call_id.expr(),
            ),
            (CallContextFieldTag::TxId, tx_id.expr()),
            (CallContextFieldTag::Depth, depth.expr() + 1.expr()),
            (CallContextFieldTag::CallerAddress, caller_address.expr()),
            (CallContextFieldTag::CalleeAddress, callee_address),
            (CallContextFieldTag::CallDataOffset, cd_address.offset()),
            (CallContextFieldTag::CallDataLength, cd_address.length()),
            (CallContextFieldTag::ReturnDataOffset, rd_address.offset()),
            (CallContextFieldTag::ReturnDataLength, rd_address.length()),
            (CallContextFieldTag::Value, value.expr()),
            (CallContextFieldTag::IsSuccess, is_success.expr()),
            (CallContextFieldTag::IsStatic, is_static.expr()),
            (CallContextFieldTag::LastCalleeId, 0.expr()),
            (CallContextFieldTag::LastCalleeReturnDataOffset, 0.expr()),
            (CallContextFieldTag::LastCalleeReturnDataLength, 0.expr()),
        ] {
            cb.call_context_lookup(
                false.expr(),
                Some(callee_call_id.expr()),
                field_tag,
                value,
            );
        }

        // Give gas stipend if value is not zero
        let callee_gas_left = callee_gas_left + has_value * 2300.expr();

        cb.require_step_state_transition(StepStateTransition {
            rw_counter: Delta(44.expr()),
            call_id: To(callee_call_id.expr()),
            is_root: To(false.expr()),
            is_create: To(false.expr()),
            code_source: To(callee_code_hash.expr()),
            gas_left: To(callee_gas_left),
            state_write_counter: To(2.expr()),
            ..StepStateTransition::new_context()
        });

        Self {
            opcode,
            tx_id,
            rw_counter_end_of_reversion,
            is_persistent,
            caller_address,
            is_static,
            depth,
            gas: gas_word,
            callee_address: callee_address_word,
            value,
            is_success,
            gas_is_u64,
            is_warm_access,
            is_warm_access_prev,
            callee_rw_counter_end_of_reversion,
            callee_is_persistent,
            value_is_zero,
            cd_address,
            rd_address,
            memory_expansion,
            transfer,
            callee_nonce,
            callee_code_hash,
            callee_nonce_is_zero,
            callee_balance_is_zero,
            callee_code_hash_is_empty,
            one_64th_gas,
            capped_callee_gas_left,
        }
    }

    fn assign_exec_step(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        block: &Block<F>,
        _: &Transaction,
        _: &Call,
        step: &ExecStep,
    ) -> Result<(), Error> {
        let [tx_id, rw_counter_end_of_reversion, is_persistent, caller_address, is_static, depth, callee_rw_counter_end_of_reversion, callee_is_persistent] =
            [
                step.rw_indices[0],
                step.rw_indices[1],
                step.rw_indices[2],
                step.rw_indices[3],
                step.rw_indices[4],
                step.rw_indices[5],
                step.rw_indices[15],
                step.rw_indices[16],
            ]
            .map(|idx| block.rws[idx].call_context_value());
        let [gas, callee_address, value, cd_offset, cd_length, rd_offset, rd_length, is_success] =
            [
                step.rw_indices[6],
                step.rw_indices[7],
                step.rw_indices[8],
                step.rw_indices[9],
                step.rw_indices[10],
                step.rw_indices[11],
                step.rw_indices[12],
                step.rw_indices[13],
            ]
            .map(|idx| block.rws[idx].stack_value());
        let (is_warm_access, is_warm_access_prev) =
            block.rws[step.rw_indices[14]].tx_access_list_value_pair();
        let [caller_balance_pair, callee_balance_pair, (callee_nonce, _), (callee_code_hash, _)] =
            [
                step.rw_indices[17],
                step.rw_indices[18],
                step.rw_indices[19],
                step.rw_indices[20],
            ]
            .map(|idx| block.rws[idx].account_value_pair());

        let opcode = step.opcode.unwrap();
        self.opcode
            .assign(region, offset, Some(F::from(opcode.as_u64())))?;

        self.tx_id
            .assign(region, offset, Some(F::from(tx_id.low_u64())))?;
        self.rw_counter_end_of_reversion.assign(
            region,
            offset,
            Some(F::from(rw_counter_end_of_reversion.low_u64())),
        )?;
        self.is_persistent.assign(
            region,
            offset,
            Some(F::from(is_persistent.low_u64())),
        )?;
        self.caller_address.assign(
            region,
            offset,
            caller_address.to_scalar(),
        )?;
        self.is_static.assign(
            region,
            offset,
            Some(F::from(is_static.low_u64())),
        )?;
        self.depth
            .assign(region, offset, Some(F::from(depth.low_u64())))?;

        self.gas.assign(region, offset, Some(gas.to_le_bytes()))?;
        self.callee_address.assign(
            region,
            offset,
            Some(callee_address.to_le_bytes()),
        )?;
        self.value
            .assign(region, offset, Some(value.to_le_bytes()))?;
        self.is_success.assign(
            region,
            offset,
            Some(F::from(is_success.low_u64())),
        )?;
        self.gas_is_u64.assign(
            region,
            offset,
            sum::value(&gas.to_le_bytes()[N_BYTES_GAS..]),
        )?;
        self.is_warm_access.assign(
            region,
            offset,
            Some(F::from(is_warm_access as u64)),
        )?;
        self.is_warm_access_prev.assign(
            region,
            offset,
            Some(F::from(is_warm_access_prev as u64)),
        )?;
        self.callee_rw_counter_end_of_reversion.assign(
            region,
            offset,
            Some(F::from(callee_rw_counter_end_of_reversion.low_u64())),
        )?;
        self.callee_is_persistent.assign(
            region,
            offset,
            Some(F::from(callee_is_persistent.low_u64())),
        )?;
        let value_is_zero = self.value_is_zero.assign(
            region,
            offset,
            sum::value(&value.to_le_bytes()),
        )?;
        let cd_address = self.cd_address.assign(
            region,
            offset,
            cd_offset,
            cd_length,
            block.randomness,
        )?;
        let rd_address = self.rd_address.assign(
            region,
            offset,
            rd_offset,
            rd_length,
            block.randomness,
        )?;
        let (_, memory_expansion_gas_cost) = self.memory_expansion.assign(
            region,
            offset,
            step.memory_size,
            [cd_address, rd_address],
        )?;
        self.transfer.assign(
            region,
            offset,
            caller_balance_pair,
            callee_balance_pair,
            value,
        )?;
        self.callee_nonce
            .assign(region, offset, callee_nonce.to_scalar())?;
        self.callee_code_hash.assign(
            region,
            offset,
            Some(Word::random_linear_combine(
                callee_code_hash.to_le_bytes(),
                block.randomness,
            )),
        )?;
        let callee_nonce_is_zero = self.callee_nonce_is_zero.assign(
            region,
            offset,
            F::from(callee_nonce.low_u64()),
        )?;
        let callee_balance_is_zero = self.callee_balance_is_zero.assign(
            region,
            offset,
            sum::value(&callee_balance_pair.1.to_le_bytes()),
        )?;
        let callee_code_hash_is_empty = self.callee_code_hash_is_empty.assign(
            region,
            offset,
            Word::random_linear_combine(
                callee_code_hash.to_le_bytes(),
                block.randomness,
            ),
            Word::random_linear_combine(EMPTY_HASH, block.randomness),
        )?;
        let is_cold_access = !is_warm_access_prev;
        let is_account_empty = callee_nonce_is_zero == F::one()
            && callee_balance_is_zero == F::one()
            && callee_code_hash_is_empty == F::one();
        let has_value = value_is_zero != F::one();
        let gas_cost = GasCost::WARM_ACCESS.as_u64()
            + if is_cold_access {
                GasCost::EXTRA_COLD_ACCESS_ACCOUNT.as_u64()
            } else {
                0
            }
            + if is_account_empty {
                GasCost::CALL_EMPTY_ACCOUNT.as_u64()
            } else {
                0
            }
            + if has_value {
                GasCost::CALL_WITH_VALUE.as_u64()
            } else {
                0
            }
            + memory_expansion_gas_cost;
        let gas_available = step.gas_left - gas_cost;
        self.one_64th_gas
            .assign(region, offset, gas_available as u128)?;
        self.capped_callee_gas_left.assign(
            region,
            offset,
            F::from(gas.low_u64()),
            F::from(gas_available - gas_available / 64),
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::evm_circuit::{
        test::run_test_circuit_incomplete_fixed_table, witness::build_block,
    };
    use bus_mapping::{
        bytecode,
        eth_types::{Address, ToWord, Transaction, Word},
        external_tracer::Account,
    };
    use std::default::Default;

    #[derive(Clone, Copy, Debug, Default)]
    struct CallContext {
        rw_counter_end_of_reversion: usize,
        is_persistent: bool,
        is_static: bool,
        gas_left: u64,
        memory_size: u32,
        state_write_counter: usize,
    }

    #[derive(Clone, Copy, Debug, Default)]
    struct Stack {
        gas: Word,
        value: Word,
        cd_offset: Word,
        cd_length: Word,
        rd_offset: Word,
        rd_length: Word,
    }

    fn test_ok(caller: Account, callee: Account, stack: Stack) {
        let bytecode = bytecode! {
            PUSH32(stack.rd_length)
            PUSH32(stack.rd_offset)
            PUSH32(stack.cd_length)
            PUSH32(stack.cd_offset)
            PUSH32(stack.value)
            PUSH32(callee.address.to_word())
            PUSH32(stack.gas)
            CALL
            STOP
        };
        let block = build_block(
            &[
                Account {
                    code: bytecode.to_vec().into(),
                    ..caller
                },
                callee,
            ],
            Transaction {
                to: Some(caller.address),
                gas: 58000.into(),
                ..Default::default()
            },
        );
        assert_eq!(run_test_circuit_incomplete_fixed_table(block), Ok(()));
    }

    #[test]
    fn call_gadget_simple() {
        let one_hundred_ether = Word::from(10).pow(20.into());
        let one_ether = Word::from(10).pow(18.into());
        let caller = Account {
            address: Address::repeat_byte(0xfe),
            balance: one_hundred_ether,
            ..Default::default()
        };
        let callee = Account {
            address: Address::repeat_byte(0xff),
            code: [0].into(),
            ..Default::default()
        };
        test_ok(
            caller,
            callee,
            Stack {
                value: one_ether,
                ..Default::default()
            },
        )
    }
}
