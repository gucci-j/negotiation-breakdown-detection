import "./Explained.css"

import {
    Area,
    AreaChart,
    CartesianGrid,
    Legend,
    ReferenceLine,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts"
import { Button, Card } from "antd"

import H from "../../components/Heading"
import P from "../../components/Paragraph"
import Checkbox from "../../components/Checkbox"
import React from "react"
import ScoreBadge from "../../components/ScoreBadge"
import calcReward from "../../utils/calcReward"

const CustomTooltip = ({ active, payload, label }) => (
    <Card>
        <div style={{ margin: "-1rem" }}>
            Score: {active ? label : 100}pts
            <br />
            Reward: $
            {active ? parseInt(payload[0].value * 100) / 100 : calcReward(100)}
        </div>
    </Card>
)

class Explained extends React.Component {
    state = {
        instructionChecked: false,
        myBestChecked: false,
        proposeChecked: false,
    }

    onInstructionChecked = e => {
        this.setState({
            instructionChecked: e.target.checked,
        })
    }

    onMyBestChecked = e => {
        this.setState({
            myBestChecked: e.target.checked,
        })
    }

    onProposeChecked = e => {
        this.setState({
            proposeChecked: e.target.checked,
        })
    }

    render() {
        let workerId
        if (
            window.location.search.split("?") &&
            window.location.search.split("?")[1] &&
            window.location.search.split("?")[1].split("&")
        ) {
            workerId = window.location.search
                .split("?")[1]
                .split("&")
                .find(q => q.split("=")[0] === "workerId")
                .split("=")[1]
        }
        const assignmentId = window.assignmentId && window.assignmentId.value
        const { instructionChecked, myBestChecked, proposeChecked } = this.state

        return (
            <div className="Explained">
                <Card className="Explained-card">
                    <H title>What is this HIT?</H>
                    <h3>
                        You will negotiate as a recruiter or worker, and get a score as
                        high as you can!
                    </h3>
                    <Card>
                        <img src="https://i.imgur.com/4TkzbBo.png" />
                        <div style={{ padding: "0 1rem" }}>
                            <h2>1. Confirm what is important for you</h2>
                            <P>
                                Issues each participant should care about are
                                different. Please confirm what issues are important to you.
                            </P>
                            <ScoreBadge weight={0.31} /> ~ : Very important
                            <br />
                            <ScoreBadge weight={0.21} /> ~{" "}
                            <ScoreBadge weight={0.3} /> : Important
                            <br />
                            <ScoreBadge weight={0.11} /> ~{" "}
                            <ScoreBadge weight={0.2} /> : Not so important
                            <br />
                            <ScoreBadge weight={0} /> ~{" "}
                            <ScoreBadge weight={0.1} /> : Not important
                            <br />
                        </div>
                    </Card>
                    <Card>
                        <img src="https://i.imgur.com/TFFVhoF.png" />
                        <div style={{ padding: "0 1rem" }}>
                            <h2>2. Talk and negotiate</h2>
                            <P>
                                Negotiate with the opponent player, and form an
                                agreement.
                            </P>
                        </div>
                    </Card>
                    <Card>
                        <img src="https://i.imgur.com/dFwRmrs.png" />
                        <div style={{ padding: "0 1rem" }}>
                            <h2>3. Propose</h2>
                            <P>
                                Propose a solution. If the opponent player
                                accepts it, the negotiation succeeded!
                                <br />
                                <span style={{ color: "#1890ff" }}>
                                    NOTICE: You can propose only <b>3</b> times!
                                    <br />
                                    Be careful!
                                </span>
                            </P>
                        </div>
                    </Card>
                    <H title>Reward</H>
                    <H header>Basic Reward</H>
                    <P>$0.2 / HIT</P>
                    <H header>Bonus</H>
                    <P>
                        If the score is higher than 50, we will pay you a bonus
                        as below chart!
                        <br />
                        If the score reaches 100, the bonus becomes $1.00!
                    </P>
                    <AreaChart
                        width={400}
                        height={200}
                        data={Array.from(
                            { length: 1001 },
                            (v, k) => k / 10,
                        ).map(score => ({
                            name: score,
                            reward: calcReward(score),
                        }))}
                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                        <XAxis
                            type="number"
                            dataKey="name"
                            name="Score"
                            unit=" pts."
                        />
                        <YAxis
                            type="number"
                            name="Reward"
                            tickFormatter={tickItem => `$${tickItem}`}
                        />
                        <CartesianGrid strokeDasharray="3 3" />
                        <Tooltip content={<CustomTooltip />} />
                        <Area dataKey="reward" fill="#3badff" />
                    </AreaChart>
                    <div>
                        <Checkbox onChange={this.onInstructionChecked}>
                            I have read the instruction
                        </Checkbox>
                    </div>
                    <div>
                        <Checkbox onChange={this.onMyBestChecked}>
                            I will try my best to <b>talk to the opponent player</b> and{" "}
                            <b>earn as high score as possible</b>
                        </Checkbox>
                    </div>
                    <div>
                        <Checkbox onChange={this.onProposeChecked}>
                            I have understood I can propose only <b>3</b> times
                        </Checkbox>
                    </div>
                    <Button
                        type="primary"
                        size="large"
                        onClick={() => {
                            this.props.forwardPage(assignmentId, workerId)
                        }}
                        disabled={
                            !workerId ||
                            !instructionChecked ||
                            !myBestChecked ||
                            !proposeChecked
                        }
                    >
                        Accept this HIT
                    </Button>
                </Card>
            </div>
        )
    }
}

export default Explained
