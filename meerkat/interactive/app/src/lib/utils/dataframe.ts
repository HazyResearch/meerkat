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
