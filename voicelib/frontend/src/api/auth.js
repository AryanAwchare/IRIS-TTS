import client from './client'

export const authApi = {
  register: (email, password) =>
    client.post('/auth/register', { email, password }).then((r) => r.data),

  login: (email, password) =>
    client.post('/auth/login', { email, password }).then((r) => r.data),

  me: () => client.get('/auth/me').then((r) => r.data),
}
