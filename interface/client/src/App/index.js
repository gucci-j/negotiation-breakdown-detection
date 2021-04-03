import { Provider, connect } from 'react-redux'
import {
  applyMiddleware,
  combineReducers,
  createStore,
} from 'redux'
import { emit, init as websocketInit } from '../websocket'

import App from './App'
import React from 'react'
import messagesReducer from '../reduxes/messages/reducers'
import optionsReducer from '../reduxes/options/reducers'
import pagesReducer from '../reduxes/pages/reducers'
import resultReducer from '../reduxes/result/reducers'
import roleReducer from '../reduxes/role/reducers'
import thunkMiddleware from 'redux-thunk'
import utilitiesReducer from '../reduxes/utilities/reducers'

const middlewares = [
  thunkMiddleware.withExtraArgument({ emit }),
]

const store = createStore(
  combineReducers({
    pages: pagesReducer,
    messages: messagesReducer,
    utilities: utilitiesReducer,
    role: roleReducer,
    result: resultReducer,
    options: optionsReducer,
  }),
  applyMiddleware(...middlewares),
)

websocketInit(store)

const stateToProps = (state) => ({
  currentPageName: state.pages.pageName
})

const ConnectedApp = connect(stateToProps)(App)

const ProvidedApp = () => (
  <Provider store={store}>
    <ConnectedApp />
  </Provider>
)

export default ProvidedApp
