/**
 * An object that holds a reference to a DataFrame on the backend, but does not
 * actually hold any of its data. 
 */
export interface DataFrameRef {
    refId: string;
}

export interface DataFrameSchema {
    columns: Array<ColumnInfo>;
    primaryKey: string;
    nrows: number;
    id: string;
}

/**
 * Important metadata about a column in a DataFrame.
 */
export interface ColumnInfo {
    name: string;
    type: string;
    cellComponent: string;
    cellProps: any,
    cellDataProp: string,
}

/**
 * A chunk of a DataFrame holds the data for a subset of the rows and columns in a full
 * DataFrame that lives on the backend. 
 */
export class DataFrameChunk {

    columnInfos: Array<ColumnInfo>
    columns: Array<string>
    posidxs: Array<number>
    keyidxs: Array<string>
    rows: Array<Array<any>>
    fullLength: number
    primaryKey: string

    constructor(
        columnInfos: Array<ColumnInfo>,
        posidxs: Array<number>,
        rows: Array<Array<any>>,
        fullLength: number,
        primaryKey: string
    ) {
        this.columnInfos = columnInfos;
        this.columns = this.columnInfos.map((col: any) => col.name);
        this.posidxs = posidxs;
        this.rows = rows;
        this.fullLength = fullLength;
        this.primaryKey = primaryKey

        let primaryKeyIndex = this.columns.findIndex((c) => c === primaryKey);
        this.keyidxs = this.rows.map((row) => row[primaryKeyIndex])
    }

    getCell(row: number, column: string) {
        let columnIdx = this.columns.indexOf(column);
        if (columnIdx === -1) {
            throw new Error(`Column ${column} does not exist in this DataFrame`);
        }
        let columnInfo = this.columnInfos[columnIdx];
        return {
            data: this.rows[row][this.columns.indexOf(column)],
            cellComponent: columnInfo.cellComponent,
            cellProps: columnInfo.cellProps,
            cellDataProp: columnInfo.cellDataProp,
            column: column
        }
    }
    getColumn(column: string) {
        let columnIdx = this.columns.indexOf(column);
        let columnInfo = this.columnInfos[columnIdx];
        console.log(column)
        return {
            data: this.rows.map((row) => row[columnIdx]),
            cellComponent: columnInfo.cellComponent,
            cellProps: columnInfo.cellProps,
            cellDataProp: columnInfo.cellDataProp,
            column: column
        }
    }
    getRow(rowIdx: number) {
        let result: any = {};
        for (let i = 0; i < this.columns.length; i++) {
            result[this.columns[i]] = this.rows[rowIdx][i]
        }
        return result

    }
    getRows() {
        let result: Array<any> = [];
        for (let i = 0; i < this.rows.length; i++) {
            result.push(this.getRow(i));
        }
        return result
    }

    length() {
        return this.rows.length;
    }

}
