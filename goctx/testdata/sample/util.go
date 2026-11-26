package sample

// formatMsg is a same-file helper. A change to Greet that references it must
// surface it as out-of-diff context even though it lives in this very file —
// the reviewer is shown only the patch, not the whole file.
func formatMsg(s string) string {
	return "[" + s + "]"
}

func Greet(name string) string {
	return formatMsg(name)
}
