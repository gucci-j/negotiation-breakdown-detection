import './Heading.css'

import React from 'react'

const Heading = ({ children, title, header, subheader, center, style }) => (
  title ?
    <h1 className='Heading' style={style}>
      {children}
    </h1> :
    header ?
      <h2 className='Heading' style={style}>
        {children}
      </h2> :
      subheader ?
        <h3 className='Heading' style={style}>
          {children}
        </h3> :
        <h4 className='Heading' style={style}>
          {children}
        </h4>

)

export default Heading