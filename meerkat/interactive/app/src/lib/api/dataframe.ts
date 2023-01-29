import { post } from '$lib/utils/requests';
/**
 * An object that holds a reference to a DataFrame on the backend, but does not
 * actually hold any of its data. 
 */
export interface DataFrameRef {
    ref_id: string;
}

export interface DataFrameSchema {
    columns: Array<ColumnInfo>;
    primary_key: string;
    nrows: number;
    id: string;
}

/**
 * Important metadata about a column in a DataFrame.
 */
export interface ColumnInfo {
    name: string;
    type: string;
    cell_component: string;
    cell_props: any,
    cell_data_prop: string,
}

/**
 * A chunk of a DataFrame holds the data for a subset of the rows and columns in a full
 * DataFrame that lives on the backend. 
 */
export class DataFrameChunk {

    column_infos: Array<ColumnInfo>
    columns: Array<string>
    posidxs: Array<number>
    keyidxs: Array<string>
    rows: Array<Array<any>>
    full_length: number
    primary_key: string

    constructor(
        column_infos: Array<ColumnInfo>,
        posidxs: Array<number>,
        rows: Array<Array<any>>,
        full_length: number,
        primary_key: string
    ) {
        this.column_infos = column_infos;
        this.columns = this.column_infos.map((col: any) => col.name);
        this.posidxs = posidxs;
        this.rows = rows;
        this.full_length = full_length;
        this.primary_key = primary_key

        let primary_key_index = this.columns.findIndex((c) => c === primary_key);
        this.keyidxs = this.rows.map((row) => row[primary_key_index])
    }

    get_cell(row: number, column: string) {
        let column_idx = this.columns.indexOf(column);
        let column_info = this.column_infos[column_idx];
        return {
            data: this.rows[row][this.columns.indexOf(column)],
            cell_component: column_info.cell_component,
            cell_props: column_info.cell_props,
            cell_data_prop: column_info.cell_data_prop,
            column: column
        }
    }
    get_column(column: string) {
        let column_idx = this.columns.indexOf(column);
        let column_info = this.column_infos[column_idx];
        return {
            data: this.rows.map((row) => row[column_idx]),
            cell_component: column_info.cell_component,
            cell_props: column_info.cell_props,
            cell_data_prop: column_info.cell_data_prop,
            column: column
        }
    }

    length() {
        return this.rows.length;
    }

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