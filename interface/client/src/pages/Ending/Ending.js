import "./Ending.css"

import { Button } from "antd"
import P from "../../components/Paragraph"
import React from "react"

class Ending extends React.Component {
    render() {
        const workerId = window.location.search
            .split("?")[1]
            .split("&")
            .find(q => q.split("=")[0] === "workerId")
            .split("=")[1]
        const { succeeded, score, reward } = this.props.result
        const { options, utilities, role } = this.props

        return (
            <div className="Ending">
                {succeeded && (
                    <div className="Ending-card">
                        <h1>The negotiation is SUCCEEDED</h1>
                        <P>
                            Your score is <b>{" " + score}</b>
                        </P>
                        <P>
                            Your reward is
                            <b>{" " + reward}</b>
                        </P>
                        <Button type="primary" htmlType="submit">
                            Submit this HIT
                        </Button>
                    </div>
                )}
                {!succeeded && (
                    <div className="Ending-card">
                        <h1>The negotiation is FAILED</h1>
                        <Button
                            type="primary"
                            onClick={() => {
                                window.location.reload()
                            }}
                        >
                            Retry
                        </Button>
                        <Button type="primary" htmlType="submit">
                            Submit this HIT
                        </Button>
                    </div>
                )}
            </div>
        )
    }
}

export default Ending
