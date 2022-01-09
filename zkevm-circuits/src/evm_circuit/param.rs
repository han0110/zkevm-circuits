// Step dimension
pub(crate) const STEP_WIDTH: usize = 32;
pub(crate) const STEP_HEIGHT: usize = 16;
pub(crate) const N_CELLS_STEP_STATE: usize = 10;

/// Maximum number of bytes that an integer can fit in field without wrapping
/// around.
pub(crate) const MAX_N_BYTES_INTEGER: usize = 31;

// Number of bytes an u64 has.
pub(crate) const N_BYTES_U64: usize = 8;

pub(crate) const N_BYTES_GAS: usize = N_BYTES_U64;

pub(crate) const N_BYTES_ACCOUNT_ADDRESS: usize = 20;

// Number of bytes that will be used of the memory address and size.
// If any of the other more signficant bytes are used it will always result in
// an out-of-gas error.
pub(crate) const N_BYTES_MEMORY_ADDRESS: usize = 5;
pub(crate) const N_BYTES_MEMORY_SIZE: usize = 4;

// Number of bytes that will be used of prorgam counter. Although the maximum
// size of execution bytecode could be at most 128kB due to the size limit of a
// transaction, which could be covered by 3 bytes, we still support program
// counter to u64 as go-ethereum in case transaction size is allowed larger in
// the future.
pub(crate) const N_BYTES_PROGRAM_COUNTER: usize = N_BYTES_U64;

// Hash output in little-endian of keccak256 with empty input.
pub(crate) const EMPTY_HASH: [u8; 32] = [
    0x70, 0xa4, 0x85, 0x5d, 0x04, 0xd8, 0xfa, 0x7b, 0x3b, 0x27, 0x82, 0xca,
    0x53, 0xb6, 0x00, 0xe5, 0xc0, 0x03, 0xc7, 0xdc, 0xb2, 0x7d, 0x7e, 0x92,
    0x3c, 0x23, 0xf7, 0x86, 0x01, 0x46, 0xd2, 0xc5,
];
