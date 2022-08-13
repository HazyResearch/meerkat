import { post } from '$lib/utils/requests';

export interface ColumnInfo {
    name: string;
    type: string;
    cell_component: string;
    cell_props: any
}

export interface DataPanelSchema {
    columns: Array<ColumnInfo>;
    id: string;
}
export interface DataPanelRows {
    column_infos?: Array<ColumnInfo>
    indices?: Array<number>
    rows?: Array<Array<any>>
    full_length?: number
}

export class FilterCriterion {
    constructor(readonly column: string, readonly value: any, readonly op: string) { }
}

export class MatchCriterion {
    constructor(readonly column: string, readonly query: string) { }
}

export async function get_schema(
    api_url: string, datapanel_id: string, columns: Array<string> | null = null
): Promise<DataPanelSchema> {
    return await post(`${api_url}/dp/${datapanel_id}/schema`, { columns: columns });
}

export async function get_rows(
    api_url: string,
    datapanel_id: string,
    start?: number,
    end?: number,
    indices?: Array<number>,
    columns?: Array<string>
): Promise<DataPanelRows> {
    return await post(
        `${api_url}/dp/${datapanel_id}/rows`,
        { start: start, end: end, indices: indices, columns: columns }
    );
}


export async function match(
    api_url: string, datapanel_id: string, match_criterion: MatchCriterion
): Promise<DataPanelSchema> {
    console.log(`${api_url}/dp/${datapanel_id}/match`)
    return await post(`${api_url}/dp/${datapanel_id}/match`, { query: match_criterion.query, input: match_criterion.column });
}

export async function sort(
    api_url: string, datapanel_id: string, by: string
): Promise<DataPanelSchema> {
    return await post(`${api_url}/dp/${datapanel_id}/sort`, { by: by });
}

export async function filter(
    api_url: string, datapanel_id: string, filter_criteria: Array<FilterCriterion>
): Promise<DataPanelSchema> {
    const columns: Array<string> = filter_criteria.map(criterion => criterion.column);
    const values: Array<any> = filter_criteria.map(criterion => criterion.value);
    const ops: Array<string> = filter_criteria.map(criterion => criterion.op);
    return await post(`${api_url}/dp/${datapanel_id}/filter`, { columns: columns, values: values, ops: ops });
}