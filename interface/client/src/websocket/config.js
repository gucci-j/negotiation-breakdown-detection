const uri = 'http://localhost:8080/'
const messageTypes = [
  'joinLobby',
  'joinRoom',
  'leaveRoom',
  'receiveMessage',
  'receiveSolution',
  'showResult',
  'completeNegotiation',
  'terminateNegotiation',
].reduce((accum, msg) => {
  accum[msg] = msg
  return accum
}, {})

export {
  uri,
  messageTypes
}