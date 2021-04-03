import "./DiscreteIssue.css"

import Heading from "../../Heading"
import { Radio } from "antd"
import React from "react"
import ScoreBadge from "../../ScoreBadge"
import { connect } from "react-redux"

const RadioGroup = Radio.Group

const styles = {
    radio: {
        display: "block",
    },
    inner: {},
}

const DiscreteIssue = ({
    title,
    weight,
    options,
    handleIssueChange,
    relatedTo,
    utilities,
    selectedOptions,
}) => {
    const relatedIssue = utilities.find(u => u.name === relatedTo)
    const relatedSelectedOption =
        selectedOptions[Object.keys(selectedOptions).find(u => u === relatedTo)]
    // options = options.map(o => {
    //     o.weight = o.bias[relatedOption]
    //     return o
    // })

    return (
        <div className="Issue">
            <Heading>
                Importance:
                <ScoreBadge
                    weight={weight + relatedIssue.weight}
                    withText={true}
                />
                <br />
                NOTICE: The importance of each issue for you is different from that
                for the opponent player.
            </Heading>
            <div style={styles.inner}>
                <div style={{ paddingLeft: 124, display: "flex" }}>
                    {relatedIssue.options.map(ro => (
                        <div key={ro.name} style={{ width: 100 }}>
                            {ro.name}
                        </div>
                    ))}
                </div>
                <RadioGroup
                    defaultValue={options[0].name}
                    onChange={e => handleIssueChange(title, e.target.value)}
                >
                    {options.map(option => (
                        <div
                            style={styles.radio}
                            key={option.name}
                            value={option.name}
                        >
                            <div className="radioInner">
                                <div className="radioInner-name">
                                    {option.name}
                                </div>
                                {relatedIssue.options.map(ro => {
                                    const biasedWeight =
                                        option.biasedWeights[ro.name]
                                    const isSelected =
                                        relatedSelectedOption === ro.name &&
                                        option.name === selectedOptions[title]
                                    const score =
                                        biasedWeight * weight +
                                        relatedIssue.weight * ro.weight
                                    return (
                                        <div
                                            key={option.name + "-" + ro.name}
                                            onClick={() => {
                                                handleIssueChange(
                                                    title,
                                                    option.name,
                                                )
                                                handleIssueChange(
                                                    relatedIssue.name,
                                                    ro.name,
                                                )
                                            }}
                                            style={{
                                                background: `rgba(24, 144, 255, ${parseInt(
                                                    score * 100,
                                                ) / 100})`,
                                                width: 100,
                                                padding: isSelected
                                                    ? "calc(0.3rem - 2px)"
                                                    : "0.3rem",
                                                border:
                                                    isSelected &&
                                                    "2px solid red",
                                                cursor: "pointer",
                                            }}
                                        >
                                            {parseInt(score * 100)}
                                        </div>
                                    )
                                })}
                            </div>
                        </div>
                    ))}
                </RadioGroup>
            </div>
        </div>
    )
}

const mapStateToProps = state => ({
    utilities: state.utilities.toJS(),
    selectedOptions: state.options.toJS(),
})

export default connect(
    mapStateToProps,
    null,
)(DiscreteIssue)
