// A callback for refreshing the current page.
export type RefreshCallback = {
    (): Promise<any>;
}

export type NoArgCallback = {
    (): void;
}