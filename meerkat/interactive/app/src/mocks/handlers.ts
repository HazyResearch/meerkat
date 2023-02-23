import { rest } from 'msw';

const messages = {
    columnInfos: [
        {
            name: "id",
            type: "number",
            cellComponent: "Cell",
            cellProps: {},
            cellDataProp: "data"
        },
        {
            name: "message",
            type: "string",
            cellComponent: "Cell",
            cellProps: {},
            cellDataProp: "data"
        },
        {
            name: "name",
            type: "string",
            cellComponent: "Cell",
            cellProps: {},
            cellDataProp: "data"
        },
        {
            name: "time",
            type: "string",
            cellComponent: "Cell",
            cellProps: {},
            cellDataProp: "data"
        },
        {
            name: "sender",
            type: "string",
            cellComponent: "Cell",
            cellProps: {},
            cellDataProp: "data"
        }
    ],
    posidxs: [0, 1, 2, 3],
    rows: [
        [0, "hello", "chatbot", "2021-01-01 00:00:00"],
        [1, "hi", "user", "2021-01-01 00:00:00"],
        [2, "how can i help", "chatbot", "2021-01-01 00:00:00"],
        [3, "im good thanks", "user", "2021-01-01 00:00:00"],
    ],
    fullLength: 4,
    primaryKey: "id"
}

const schema = {
    id: "mock",
    columns: messages.columnInfos,
    primaryKey: "id",
    nrows: 4
}


// Define handlers that catch the corresponding requests and return the mock data.
export const handlers = [
    rest.post(`http://test.app/df/mock/rows/`, (req, res, ctx) => {
        return res(ctx.status(200), ctx.json(messages))
    }),

    rest.post(`http://test.app/df/mock/schema/`, (req, res, ctx) => {
        return res(ctx.status(200), ctx.json(schema))
    }),

    rest.get(`http://test.app/api/`, (req, res, ctx) => {
        return res(ctx.status(200), ctx.json({ "hello": "world" }))
    }),
];