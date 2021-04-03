import * as actions from './actions'

import { messageTypes, uri } from './config'

import io from 'socket.io-client'

const socket = io(uri)

const init = (store) => {
  Object
    .keys(messageTypes)
    .forEach(
      type => socket.on(type, (payload) =>
        store.dispatch({
          type,
          payload: { ...payload, socket }
        })
      )
    )
}

const emit = (type, payload) => socket.emit(type, payload)

export {
  init,
  emit,
  socket,
  actions,
}
