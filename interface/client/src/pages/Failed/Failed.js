import "./Failed.css"

import { Button, Card } from "antd"

import H from "../../components/Heading"
import P from "../../components/Paragraph"
import React from "react"

class Failed extends React.Component {
    render() {
        return (
            <div className="Failed">
                <Card className="Failed-card">
                    <H title>We're sorry.</H>
                    <P>
                        The opponent player disconnected. <br />
                        <b>Please reload this page and retry this HIT.</b>
                    </P>
                </Card>
            </div>
        )
    }
}

export default Failed
