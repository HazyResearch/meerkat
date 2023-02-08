export interface CellInterface {
    data: any;
    column?: string;
    cellComponent?: string;
    cellProps?: object;
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
    endpointId: string;
};

export interface PageType {
    name: string;
    component: ComponentType
};
