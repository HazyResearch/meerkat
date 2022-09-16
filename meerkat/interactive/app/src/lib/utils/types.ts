import type { Writable } from "svelte/store";
import type { Dictionary } from "underscore";

export interface Component {
    component_id: string;
    name: string;
    props: any; 
}

export interface Layout {
    name: string
    props: any;
}

export interface Interface {
    name: string;
    layout: Layout;
    components: Array<Component> | any 
}

export interface EditTarget {
    target: Writable<Box>
    target_id_column: string
    source_id_column: string
}

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