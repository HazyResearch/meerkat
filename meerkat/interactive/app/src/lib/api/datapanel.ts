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

export interface FilterCriterion {
    is_enabled: boolean;
    column: string;
    op: string;
    value: any;
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
    console.log(`${api_url}/box/${datapanel_id}/rows`);
    console.log(`start: ${start}`);
    console.log(`end: ${end}`);
    console.log(`indices: ${indices}`);
    console.log(`columns: ${columns}`);

    return await post(
        `${api_url}/box/${datapanel_id}/rows`,
        { start: start, end: end, indices: indices, columns: columns }
    );
}


export async function match(
    api_url: string, datapanel_id: string, match_criterion: MatchCriterion
): Promise<DataPanelSchema> {
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

export async function filter_box(
    api_url: string, 
    box_id: string, 
    filter_criteria: Array<FilterCriterion>
): Promise<DataPanelSchema> {
    const columns: Array<string> = filter_criteria.map(criterion => criterion.column);
    const values: Array<any> = filter_criteria.map(criterion => criterion.value);
    const ops: Array<string> = filter_criteria.map(criterion => criterion.op);
    return await post(`${api_url}/box/${box_id}/filter`, { columns: columns, values: values, ops: ops });
}

export async function undo_box(
    api_url: string, 
    box_id: string, 
    operation_id: string
): Promise<DataPanelSchema> {
    return await post(`${api_url}/box/${box_id}/undo`, { operation_id: operation_id });
}