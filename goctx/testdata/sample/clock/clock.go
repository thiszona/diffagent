package clock

// Ranger checks whether a time falls in a range.
type Ranger interface {
	InRange(check, start, end int) bool
}

type ranger struct{}

// New returns the default Ranger.
func New() Ranger { return &ranger{} }

// InRange reports whether check is within [start, end].
func (r *ranger) InRange(check, start, end int) bool {
	return check >= start && check <= end
}

// Add sums two times.
func Add(a, b int) int {
	return a + b
}
