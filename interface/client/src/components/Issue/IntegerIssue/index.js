import { Area, AreaChart, ReferenceLine, XAxis, YAxis } from "recharts"
import { InputNumber, Slider } from "antd"

import Heading from "../../Heading"
import React from "react"
import ScoreBadge from "../../ScoreBadge"
import { connect } from "react-redux"

class IntegerIssue extends React.Component {
    onChange = value => {
        const { title, handleIssueChange } = this.props

        handleIssueChange(title, value)
    }

    render() {
        const {
            title,
            weight,
            min,
            max,
            formatter,
            handleIssueChange,
            role,
            value,
        } = this.props
        const marks = []
        marks[min] = formatter(min)
        marks[max] = formatter(max)
        const calcScore = (role, value) =>
            role === "recruiter"
                ? (max - value) / (max - min)
                : (value - min) / (max - min)
        const chartData = Array.from(
            { length: max - min + 1 },
            (_, k) => k + min,
        ).map(v => ({
            name: v,
            score: calcScore(role, v) * parseInt(weight * 100),
        }))

        return (
            <div className="Issue">
                <Heading>
                    Importance:
                    <ScoreBadge weight={weight} withText={true} />
                    <br />
                    NOTICE: The importance of each issue for you is different from
                    that for the opponent player.
                </Heading>
                <div style={{ display: "flex" }}>
                    <div style={{ width: 400, paddingLeft: 55 }}>
                        <Slider
                            min={min}
                            max={max}
                            marks={marks}
                            value={value}
                            onChange={this.onChange}
                            tipFormatter={formatter}
                        />
                    </div>
                    <div>
                        <InputNumber
                            min={min}
                            max={max}
                            style={{ marginLeft: 16 }}
                            value={value}
                            onChange={this.onChange}
                            onKeyDown={e =>
                                e.keyCode === 13 && e.preventDefault()
                            }
                        />
                    </div>
                </div>
                <div style={{ display: "flex" }}>
                    <AreaChart width={400} height={100} data={chartData}>
                        <XAxis dataKey="name" />
                        <YAxis />
                        <ReferenceLine x={value} stroke="red" label="" />
                        <Area
                            type="monotone"
                            dataKey="score"
                            stroke="#1890ff"
                            fill="#1890ff"
                        />
                    </AreaChart>
                </div>
            </div>
        )
    }
}

const mapStateToProps = (state, ownProps) => ({
    role: state.role,
    value: state.options.get(ownProps.title),
    options: state.options,
})

export default connect(
    mapStateToProps,
    null,
)(IntegerIssue)
