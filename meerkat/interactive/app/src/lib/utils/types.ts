import type { Writable } from "svelte/store";

export interface CellInterface {
    data: any;
    column?: string;
    cell_component?: string;
    cell_props?: object;
}

export interface ComponentType {
    component_id: string;
    path: string;
    name: string;
    props: any;
    slots: any;
    library: string;
};

export interface Endpoint {
    endpoint_id: string;
};

export interface PageType {
    name: string;
    component: ComponentType
};

export interface EditTarget {
    target: Writable<any>
    target_id_column: string
    source_id_column: string
};
