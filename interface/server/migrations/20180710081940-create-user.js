'use strict';
module.exports = {
  up: (queryInterface, Sequelize) => {
    return queryInterface.createTable('users', {
      id: {
        allowNull: false,
        autoIncrement: true,
        primaryKey: true,
        type: Sequelize.INTEGER
      },
      socket_id: {
        type: Sequelize.STRING
      },
      assignment_id: {
        type: Sequelize.STRING
      },
      worker_id: {
        type: Sequelize.STRING
      },
      is_bonus_paid: {
        type: Sequelize.BOOLEAN
      },
      assignment_status: {
        type: Sequelize.STRING
      },
      role: {
        type: Sequelize.STRING
      },
      room_id: {
        type: Sequelize.INTEGER,
        references: {
          model: 'rooms',
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
      joined_at: {
        type: Sequelize.DATE
      },
      utilities: {
        type: Sequelize.JSON
      },
    });
  },
  down: (queryInterface, Sequelize) => {
    return queryInterface.dropTable('users');
  }
};