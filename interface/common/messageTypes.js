const messageTypeNames = [
    'joinLobby',
    'joinRoom',
    'sendMessage',
    'sendSolution',
    'acceptSolution',
    'terminateNegotiation',
]

const messageTypes = {}

messageTypeNames.map(name => {
    messageTypes[name] = name
})

module.exports = messageTypes