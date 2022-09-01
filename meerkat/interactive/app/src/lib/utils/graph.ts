
export interface Box {
    box_id: string
    type: "SliceBy" | "DataPanel"
}

export interface SliceByBox extends Box {
    type: "SliceBy"
}

export interface DataPanelBox extends Box {
    type: "DataPanel"
}