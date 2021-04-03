module.exports = {
    up: (queryInterface, Sequelize) => (
        queryInterface.createTable('solutions', {
            id: {
                allowNull: false,
                autoIncrement: true,
                primaryKey: true,
                type: Sequelize.INTEGER
            },
            body: {
                type: Sequelize.JSON
            },
            accepted: {
                type: Sequelize.BOOLEAN
            },
            user_id: {
                type: Sequelize.INTEGER,
                references: {
                    model: 'users',
                    key: 'id'
                }
            },
            created_at: {
                allowNull: false,
                type: Sequelize.DATE
            },
            updated_at: {
                allowNull: false,
                type: Sequelize.DATE
            },
            deleted_at: {
                type: Sequelize.DATE
            },
        })
    ),
    down: (queryInterface, Sequelize) => (
        queryInterface.dropTable('solutions')
    )
}
