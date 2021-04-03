import "./MessageList.css"

import { Button, Modal, Tooltip } from "antd"

import React from "react"
import calcScore from "../../utils/calcScore"

const roles = ["recruiter", "worker"]

const Icon = ({ name }) => (
    <i
        class={`fas fa-${name}`}
        style={{
            paddingRight: 5,
        }}
    />
)

class MessageList extends React.Component {
    handleAcceptClick = message => {
        const { acceptSolution, options, utilities, role } = this.props

        Modal.confirm({
            title: "Do you really accept the proposal?",
            content: <div />,
            onOk() {
                acceptSolution(message.id)
            },
            onCancel() {},
        })
    }

    render() {
        const { messages, role, utilities } = this.props

        return (
            <div className="MessageList">
                <div className="MessageList-inner">
                    <div className="MessageList-message">
                        <div className="MessageList-message-name">SYSTEM</div>
                        <div className="MessageList-message-body">
                            You are now a {" "}
                            <b className="MessageList-role">
                                {role === "worker" ? "worker" : "job recruiter"}
                            </b>
                            !
                            <br />
                            {role === "worker" ? (
                                <span>
                                    You are looking for a job with a job
                                    recruiter.
                                </span>
                            ) : (
                                <span>
                                    You are introducing the opponent player as a
                                    worker to customers.
                                </span>
                            )}
                            <br />
                            Negotiate with the {roles.find(r => r !== role)}, and
                            form a consensus.
                            <br />
                            If you think you could reach an agreement with the
                            opponent player,
                            <br />
                            press the "PROPOSE" button and send your solution.
                        </div>
                    </div>

                    {messages &&
                        messages.map(m =>
                            m.body ? (
                                <div
                                    key={m.id}
                                    className={`MessageList-message ${m.fromYourself &&
                                        "fromYourself"}`}
                                >
                                    <div className="MessageList-message-name">
                                        {m.fromYourself
                                            ? role.toUpperCase()
                                            : roles
                                                  .find(r => r !== role)
                                                  .toUpperCase()}
                                    </div>
                                    <div className="MessageList-message-body">
                                        {m.body}
                                    </div>
                                </div>
                            ) : (
                                <div
                                    key={m.id}
                                    className={`MessageList-solution ${m.fromYourself &&
                                        "fromYourself"}`}
                                >
                                    <div className="MessageList-message-body">
                                        {!m.fromYourself && (
                                            <p>
                                                The opponent player proposed a
                                                solution
                                            </p>
                                        )}
                                        {Object.keys(m.solution).map(issue => {
                                            const option = m.solution[issue]
                                            return (
                                                <div>
                                                    <b>{issue}</b>: {option}
                                                </div>
                                            )
                                        })}
                                        <div
                                            className={`line ${m.fromYourself &&
                                                "fromYourself"}`}
                                        />
                                        <div className="score">
                                            Score (for you):{" "}
                                            {parseInt(
                                                calcScore(
                                                    m.solution,
                                                    utilities,
                                                    role,
                                                ) * 100,
                                            )}
                                        </div>
                                        {!m.fromYourself && (
                                            <Tooltip
                                                placement="top"
                                                title="Accept the solution that the opponent player proposed"
                                            >
                                                <Button
                                                    type="primary"
                                                    size="large"
                                                    onClick={() => {
                                                        this.handleAcceptClick(
                                                            m,
                                                        )
                                                    }}
                                                >
                                                    <Icon name="check-circle" />
                                                    ACCEPT
                                                </Button>
                                            </Tooltip>
                                        )}
                                    </div>
                                </div>
                            ),
                        )}
                </div>
            </div>
        )
    }
}

export default MessageList
