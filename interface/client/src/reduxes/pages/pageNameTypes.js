const pageNameTypes = [
  'EXPLAINED',
  'WAITING',
  'TALKING',
  'ENDING',
  'FAILED',
].reduce((accum, msg) => {
  accum[msg] = Symbol(msg)
  return accum
}, {})

export default pageNameTypes