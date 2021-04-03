export default (sequelize, DataTypes) => {
  const comment = sequelize.define('comment', {
    body: DataTypes.STRING,
    user_id: DataTypes.INTEGER
  }, {
      underscored: true,
      paranoid: true,
    })
  comment.associate = function (models) {
    comment.belongsTo(models.user)
  }
  return comment
}