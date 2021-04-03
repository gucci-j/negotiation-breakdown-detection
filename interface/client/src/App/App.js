import './App.css'

import Ending from '../pages/Ending'
import Explained from '../pages/Explained'
import Failed from '../pages/Failed'
import React from 'react'
import Talking from '../pages/Talking'
import Waiting from '../pages/Waiting'
import pageNameTypes from '../reduxes/pages/pageNameTypes'

const { EXPLAINED, WAITING, TALKING, ENDING, FAILED } = pageNameTypes

const App = (props) => {
  switch (props.currentPageName) {
    case EXPLAINED:
      return <Explained />
    case WAITING:
      return <Waiting />
    case TALKING:
      return <Talking />
    case ENDING:
      return <Ending />
    case FAILED:
      return <Failed />
    default:
      return <div />
  }
}

export default App