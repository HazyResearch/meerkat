import { post } from '$lib/utils/requests';

export interface ColumnInfo {
    name: string;
    type: string;
    cell_component: string;
    cell_props: any
}

export interface DataFrameSchema {
    columns: Array<ColumnInfo>;
    id: string;
}
export interface DataFrameRows {
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
    source: string;
    is_fixed: boolean;
}

export interface SortCriterion {
    id: string;
    is_enabled: boolean;
    column: string;
    ascending: boolean;
}

export class MatchCriterion {
    constructor(readonly column: string, readonly query: string) { }
}

export async function match(
    api_url: string, dataframe_id: string, match_criterion: MatchCriterion
): Promise<DataFrameSchema> {
    return await post(`${api_url}/df/${dataframe_id}/match`, { query: match_criterion.query, input: match_criterion.column });
}

export async function sort(
    api_url: string, dataframe_id: string, by: string
): Promise<DataFrameSchema> {
    return await post(`${api_url}/df/${dataframe_id}/sort`, { by: by });
}

export async function filter(
    api_url: string, dataframe_id: string, filter_criteria: Array<FilterCriterion>
): Promise<DataFrameSchema> {
    const columns: Array<string> = filter_criteria.map(criterion => criterion.column);
    const values: Array<any> = filter_criteria.map(criterion => criterion.value);
    const ops: Array<string> = filter_criteria.map(criterion => criterion.op);
    return await post(`${api_url}/df/${dataframe_id}/filter`, { columns: columns, values: values, ops: ops });
}

export async function filter_ref(
    api_url: string,
    ref_id: string,
    filter_criteria: Array<FilterCriterion>
): Promise<DataFrameSchema> {
    const columns: Array<string> = filter_criteria.map(criterion => criterion.column);
    const values: Array<any> = filter_criteria.map(criterion => criterion.value);
    const ops: Array<string> = filter_criteria.map(criterion => criterion.op);
    return await post(`${api_url}/ref/${ref_id}/filter`, { columns: columns, values: values, ops: ops });
}

export async function undo_ref(
    api_url: string,
    ref_id: string,
    operation_id: string
): Promise<DataFrameSchema> {
    return await post(`${api_url}/ref/${ref_id}/undo`, { operation_id: operation_id });
}