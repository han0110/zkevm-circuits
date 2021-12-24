package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/ethereum/go-ethereum/common"

	"main/gethutil"
)

func main() {
	address := common.BytesToAddress([]byte{0xff})
	assembly := gethutil.NewAssembly().Add(0xdeadbeef, 0xcafeb0ba).Sub(0xfaceb00c, 0xb0bacafe)

	accounts := []gethutil.Account{{Address: address, Code: assembly.Bytecode()}}
	tx := gethutil.Transaction{To: &address, GasLimit: 21100}

	result, err := gethutil.TraceTx(gethutil.BlockConstant{}, accounts, tx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to trace tx, err: %v\n", err)
	}

	bytes, err := json.MarshalIndent(result.StructLogs, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to marshal logs, err: %v\n", err)
	}

	fmt.Fprintln(os.Stdout, string(bytes))
}
