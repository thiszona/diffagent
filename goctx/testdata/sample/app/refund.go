package app

import "sample/model"

// Refund names the bank a refund goes to.
func Refund(users model.UserReader, id string) string {
	u, err := users.GetUser(id)
	if err != nil {
		return ""
	}
	return u.Bank.Name
}
