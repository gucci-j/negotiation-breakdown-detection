import "./Talking.css"

import { Button, Card, Input, Layout, Modal, Tabs, Tooltip } from "antd"

import DependentDiscreteIssue from "../../components/Issue/DependentDiscreteIssue"
import DiscreteIssue from "../../components/Issue/DiscreteIssue"
import Heading from "../../components/Heading"
import IntegerIssue from "../../components/Issue/IntegerIssue"
import MessageList from "../../components/MessageList"
import React from "react"
import ScoreBadge from "../../components/ScoreBadge"
import calcScore from "../../utils/calcScore"
import calcReward from "../../utils/calcReward"
import score2emoji from "../../utils/score2emoji"
import constant from "../../utils/constant"

const TabPane = Tabs.TabPane
const { Sider, Content } = Layout

const salaryFormatter = value => `$${value} / hour`
const holidayFormatter = value => `${value} days / week`

const Icon = ({ name }) => (
    <i
        className={`fas fa-${name}`}
        style={{
            paddingRight: 5,
        }}
    />
)

const TabName = ({ name, weight }) => (
    <div>
        {name}
        <ScoreBadge
            weight={weight}
            style={{
                marginLeft: 8,
            }}
        />
    </div>
)

class Talking extends React.Component {
    state = {
        score: 0,
        scoreOpacity: 1,
        text: "",
    }

    componentDidMount() {
        const { options, utilities, role } = this.props
        const score = calcScore(options, utilities, role)
        this.setState({
            score,
        })
    }

    componentDidUpdate(prevProps, prevState) {
        const { options, utilities, role } = this.props
        if (prevProps.options !== options) {
            const score = calcScore(options, utilities, role)
            this.setState({
                score,
                scoreOpacity: 0,
            })
            this.animate = setTimeout(
                () => this.setState({ scoreOpacity: 1 }),
                1,
            )
        }
        if (prevProps.messages && prevProps.messages !== this.props.messages) {
            const messages = window.document.querySelectorAll(
                ".MessageList-message",
            )
            messages[messages.length - 1].scrollIntoView()
        }
    }

    componentWillUnmount() {
        clearTimeout(this.animate)
    }

    handleKeyDown = e => {
        if (e.keyCode === 13) {
            e.preventDefault()
            if (e.target.value.length > 0) {
                this.props.sendMessage(e.target.value)
                this.setState({
                    text: "",
                })
            }
        }
    }

    handleChange = e => {
        this.setState({
            text: e.target.value,
        })
    }

    handleSendClick = e => {
        this.props.sendMessage(this.state.text)
        this.setState({
            text: "",
        })
    }

    handleProposeClick = e => {
        e.preventDefault()

        const { sendSolution, options, utilities, messages } = this.props
        const proposals = messages.filter(m => !m.body && m.fromYourself)

        Modal.confirm({
            title: "Can you two really form an agreement?",
            content: (
                <div>
                    <Card>
                        {utilities.map(utility => (
                            <div>
                                {utility.name + ": "}
                                {utility.type === "DISCRETE" ? (
                                    <b>{options[utility.name]}</b>
                                ) : utility.name === "Salary" ? (
                                    <b>$ {options[utility.name]} / hour</b>
                                ) : utility.name === "Weekly holiday" ? (
                                    <b>{options[utility.name]} days / week</b>
                                ) : (
                                    <b>{options[utility.name]}</b>
                                )}
                            </div>
                        ))}
                        {"Your Score: "}
                        <b>
                            {parseInt(
                                calcScore(
                                    this.props.options,
                                    utilities,
                                    this.props.role,
                                ) * 1000,
                            ) / 10}
                        </b>
                    </Card>
                    {<h3>Times you sent a solution: {proposals.size} / 3</h3>}
                    {proposals.size === 2 && (
                        <h3 style={{ color: "#f44336" }}>
                            <Icon name="exclamation-triangle" />
                            This is the last chance that you can send a
                            solution.
                        </h3>
                    )}
                </div>
            ),
            onOk() {
                proposals.size < 2
                    ? sendSolution(options)
                    : Modal.confirm({
                          title: "Really?",
                          content: (
                              <h3 style={{ color: "#f44336" }}>
                                  <Icon name="exclamation-triangle" />
                                  This is the last chance that you can send a
                                  solution.
                              </h3>
                          ),
                          onOk: sendSolution(options),
                      })
            },
            onCancel() {},
        })
    }

    handleTerminateClick = e => {
        const { terminateNegotiation, options, utilities } = this.props
        e.preventDefault()
        Modal.confirm({
            title: "Do you really want to terminate the negotiation?",
            content: (
                <div>
                    <b style={{ color: "red" }}>
                        Your HITs would be rejected if you terminate the
                        negotiation.
                    </b>
                </div>
            ),
            onOk() {
                terminateNegotiation()
            },
            onCancel() {},
        })
    }

    handleIssueChange = (issue, option) => {
        this.props.changeOption(issue, option)
    }

    render() {
        const { utilities, messages } = this.props
        const { scoreOpacity } = this.state
        const proposals = messages.filter(m => !m.body && m.fromYourself)
        const score =
            parseInt(
                calcScore(this.props.options, utilities, this.props.role) *
                    1000,
            ) / 10

        return (
            <Layout className="Talking">
                <Content>
                    <Heading title>
                        <Icon name="message" />
                        Conversation
                    </Heading>
                    <MessageList />
                    <Tooltip
                        placement="topLeft"
                        title="Negotiate with chatting!"
                        defaultVisible={true}
                    >
                        <div style={{ display: "flex" }}>
                            <Input.TextArea
                                autoFocus
                                rows={1}
                                placeholder='Input a message and press the ENTER key or click the "SEND" button.'
                                value={this.state.text}
                                onChange={this.handleChange}
                                onKeyDown={this.handleKeyDown}
                            />
                            <Button
                                type="primary"
                                onClick={this.handleSendClick}
                                disabled={
                                    !(
                                        this.state.text &&
                                        this.state.text.length > 0
                                    )
                                }
                            >
                                <Icon name="paper-plane" />
                                SEND
                            </Button>
                        </div>
                    </Tooltip>
                    <div style={{ margin: "1rem 0" }}>
                        <Button.Group>
                            <Tooltip
                                placement="top"
                                title={
                                    proposals.size >= 3
                                        ? "You already proposed a solution"
                                        : messages.size < constant.minMessages
                                        ? `In order to propose a solution, please send messages more than ${
                                              constant.minMessagesText
                                          } times in a total of you and an opponent.`
                                        : "Propose the solution to the opponent player"
                                }
                            >
                                <Button
                                    type="primary"
                                    size="large"
                                    disabled={
                                        proposals.size >= 3 ||
                                        messages.size < constant.minMessages
                                    }
                                    onClick={this.handleProposeClick}
                                >
                                    <Icon name="arrow-circle-up" />
                                    PROPOSE
                                </Button>
                            </Tooltip>
                            <Tooltip
                                placement="top"
                                title="Terminate this negotiation"
                            >
                                <Button
                                    type="primary"
                                    size="large"
                                    onClick={this.handleTerminateClick}
                                >
                                    <Icon name="ban" />
                                    TERMINATE
                                </Button>
                            </Tooltip>
                        </Button.Group>
                        {messages.size < constant.minMessages && (
                            <div style={{ marginTop: "0.5rem" }}>
                                <b>
                                    In order to propose a solution,
                                    <span style={{ color: "#1194fd" }}>
                                        {` please send messages more than ${
                                            constant.minMessagesText
                                        } times `}
                                    </span>
                                    in a total of you and an opponent.
                                </b>
                            </div>
                        )}
                        {messages.size >= constant.minMessages && (
                            <div style={{ marginTop: "0.5rem" }}>
                                <b>
                                    You can propose more
                                    <span style={{ color: "#1194fd" }}>
                                        {` ${3 - proposals.size} `}
                                    </span>
                                    times
                                </b>
                            </div>
                        )}
                    </div>
                </Content>
                <Sider>
                    <Heading title>
                        <Icon name="solution" />
                        Solution
                    </Heading>
                    <div style={{ display: "flex" }}>
                        <div style={{ marginRight: "2rem" }}>
                            <Heading subheader>Score:</Heading>
                            <Heading
                                header
                                style={{
                                    color: scoreOpacity < 1 ? "#fff" : "black",
                                    transition: scoreOpacity === 1 && "all 1s",
                                }}
                            >
                                {score} / 100
                            </Heading>
                        </div>
                        <div style={{ marginRight: "2rem" }}>
                            <Heading subheader>HIT Reward:</Heading>
                            <Heading
                                header
                                style={{
                                    color: scoreOpacity < 1 ? "#fff" : "black",
                                    transition: scoreOpacity === 1 && "all 1s",
                                }}
                            >
                                ${parseInt(calcReward(score) * 100) / 100}{" "}
                            </Heading>
                        </div>
                        <div>
                            <Heading subheader>&nbsp;</Heading>
                            <Heading
                                header
                                style={{
                                    scoreOpacity: scoreOpacity,
                                    transition: scoreOpacity === 1 && "all 1s",
                                }}
                            >
                                {score2emoji(score)}
                            </Heading>
                        </div>
                    </div>
                    <div className="Issues">
                        <Tabs type="card">
                            {utilities
                                .filter(
                                    utility =>
                                        !utilities.find(
                                            u => u.relatedTo === utility.name,
                                        ),
                                )
                                .map(utility =>
                                    utility.type === "INTEGER" ? (
                                        <TabPane
                                            key={utility.name}
                                            tab={<TabName {...utility} />}
                                        >
                                            <IntegerIssue
                                                key={utility.name}
                                                title={utility.name}
                                                weight={utility.weight}
                                                handleIssueChange={
                                                    this.handleIssueChange
                                                }
                                                formatter={
                                                    utility.name === "Salary"
                                                        ? salaryFormatter
                                                        : holidayFormatter
                                                }
                                                {...utility}
                                            />
                                        </TabPane>
                                    ) : utility.relatedTo ? (
                                        <TabPane
                                            key={utility.name}
                                            tab={
                                                <TabName
                                                    name={
                                                        utility.name +
                                                        " & " +
                                                        utility.relatedTo
                                                    }
                                                    weight={
                                                        utility.weight +
                                                        utilities.find(
                                                            u =>
                                                                u.name ===
                                                                utility.relatedTo,
                                                        ).weight
                                                    }
                                                />
                                            }
                                        >
                                            <DependentDiscreteIssue
                                                key={utility.name}
                                                title={utility.name}
                                                relatedTo={utility.relatedTo}
                                                weight={utility.weight}
                                                handleIssueChange={
                                                    this.handleIssueChange
                                                }
                                                options={utility.options}
                                            />
                                        </TabPane>
                                    ) : (
                                        <TabPane
                                            key={utility.name}
                                            tab={<TabName {...utility} />}
                                        >
                                            <DiscreteIssue
                                                key={utility.name}
                                                title={utility.name}
                                                weight={utility.weight}
                                                handleIssueChange={
                                                    this.handleIssueChange
                                                }
                                                options={utility.options}
                                            />
                                        </TabPane>
                                    ),
                                )}
                        </Tabs>
                    </div>
                </Sider>
            </Layout>
        )
    }
}

export default Talking
