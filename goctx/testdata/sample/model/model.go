package model

// User is a customer.
type User struct {
	ID   string
	Bank *Bank
}

// Bank is a user's bank account.
type Bank struct {
	Name string
}

// UserReader reads users; store.Store implements it.
type UserReader interface {
	GetUser(id string) (*User, error)
}
