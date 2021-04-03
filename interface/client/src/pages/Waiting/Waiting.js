import "./Waiting.css"

import React from "react"
import { Spin } from "antd"

class Waiting extends React.Component {
    render() {
        return (
            <div className="Waiting">
                <Spin
                    className="Waiting-spin"
                    tip="Waiting for another user..."
                />
            </div>
        )
    }
}

export default Waiting
