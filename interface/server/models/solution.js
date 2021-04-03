export default (sequelize, DataTypes) => {
    const solution = sequelize.define('solution', {
        body: DataTypes.STRING,
        user_id: DataTypes.INTEGER,
        accepted: DataTypes.BOOLEAN,
    }, {
            underscored: true,
            paranoid: true,
        })
    solution.associate = function (models) {
        solution.belongsTo(models.user)
    }
    return solution
}