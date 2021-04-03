const issues = [
    {
        name: 'Salary',
        type: 'INTEGER',
        min: 20,
        max: 50
    },
    {
        name: 'Position',
        type: 'DISCRETE',
        relatedTo: 'Company',
        options: [
            {
                name: 'Engineer'
            },
            {
                name: 'Manager',
            },
            {
                name: 'Designer',
            },
            {
                name: 'Sales',
            },
        ]
    },
    {
        name: 'Weekly holiday',
        type: 'INTEGER',
        min: 2,
        max: 6
    },
    {
        name: 'Workplace',
        type: 'DISCRETE',
        options: [
            {
                name: 'Tokyo',
            },
            {
                name: 'Seoul',
            },
            {
                name: 'Beijing',
            },
            {
                name: 'Sydney',
            }
        ]
    },
    {
        name: 'Company',
        type: 'DISCRETE',
        options: [
            {
                name: 'Google',
            },
            {
                name: 'Amazon',
            },
            {
                name: 'Facebook',
            },
            {
                name: 'Apple',
            }
        ]
    },
]

export default issues