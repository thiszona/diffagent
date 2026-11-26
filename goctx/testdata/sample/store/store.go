package store

import "sample/model"

// Store keeps users in memory.
type Store struct {
	users map[string]*model.User
}

// GetUser wraps its unexported twin, as the repositories wrap theirs in
// decorator.Execute.
func (s *Store) GetUser(id string) (*model.User, error) {
	return logged("GetUser", func() (*model.User, error) {
		return s.getUser(id)
	})
}

func (s *Store) getUser(id string) (*model.User, error) {
	u := s.users[id]
	return u, nil
}

func logged[T any](name string, f func() (T, error)) (T, error) {
	_ = name
	return f()
}
