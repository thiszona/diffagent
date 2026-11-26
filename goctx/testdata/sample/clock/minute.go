package clock

type minute struct{}

// Add shares the changed function's name: its declaration line is no hit.
func (minute) Add(a, b int) int { return a + b }
