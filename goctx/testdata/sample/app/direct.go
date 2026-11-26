package app

import "sample/clock"

// Total adds two fees.
func Total(a, b int) int {
	sum := clock.Add(a, b)
	return sum
}
