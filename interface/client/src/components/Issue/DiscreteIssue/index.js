import "./DiscreteIssue.css"

import { Radio, Select } from "antd"

import Heading from "../../Heading"
import React from "react"
import ScoreBadge from "../../ScoreBadge"
import { connect } from "react-redux"

const RadioGroup = Radio.Group

const cols = {
    weight: { tickInterval: 0.2 },
}

const styles = {
    radio: {
        display: "block",
    },
    inner: {
        display: "flex",
    },
}

const DiscreteIssue = ({
    title,
    weight,
    options,
    handleIssueChange,
    selectedOption,
}) => (
    <div className="Issue">
        <Heading>
            Importance:
            <ScoreBadge weight={weight} withText={true} />
            <br />
            NOTICE: The importance of each issue for you is different from that for
            the opponent player.
        </Heading>
        <div style={styles.inner}>
            <RadioGroup
                defaultValue={selectedOption}
                value={selectedOption}
                onChange={e => handleIssueChange(title, e.target.value)}
                onKeyDown={e => e.keyCode === 13 && e.preventDefault()}
            >
                {options.map(option => (
                    <Radio
                        style={styles.radio}
                        key={option.name}
                        value={option.name}
                        onKeyDown={e => e.keyCode === 13 && e.preventDefault()}
                    >
                        <div className="radioInner">
                            <div className="radioInner-name">{option.name}</div>
                            <div
                                className="bar"
                                style={{
                                    width:
                                        option.weight * weight * 300 + 5 + "px",
                                }}
                            />
                            {parseInt(option.weight * weight * 100)}
                        </div>
                    </Radio>
                ))}
            </RadioGroup>
        </div>
    </div>
)

const mapStateToProps = (state, ownProps) => {
    const selectedOptions = state.options.toJS()
    return {
        selectedOption: selectedOptions[ownProps.title],
    }
}

export default connect(
    mapStateToProps,
    null,
)(DiscreteIssue)
