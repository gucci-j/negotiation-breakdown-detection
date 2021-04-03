const messageTypeNames = [
    'joinRoom',
]

const messageTypes = {}

messageTypeNames.map(name => (
    messageTypes[name] = Symbol(name)
))

export default messageTypes