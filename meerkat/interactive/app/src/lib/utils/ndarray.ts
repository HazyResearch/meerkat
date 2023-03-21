export class NdArray {
    // A flattened C-contiguous array of the data.
    readonly data: Array<number>;
    readonly shape: Array<number>;

    constructor(data: Array<number>, shape: Array<number>) {
        this.data = data;
        this.shape = shape;
    }

    public slices(dim: number): Array<NdArray> {
        const sliceShape = this.shape.filter((_, i) => i !== dim);
        const sliceSize = sliceShape.reduce((accumulator, currentValue) => accumulator * currentValue, 1);
        const numSlices = this.shape[dim];

        // Create empty arrays representing the slices.
        const slicesData = new Array(this.shape[dim]);

        // When dimension is 0, we can shortcut the return.
        if (dim === 0) {
            for (let i = 0; i < this.shape[dim]; i++) {
                slicesData[i] = this.data.slice(i * sliceSize, (i + 1) * sliceSize);
            }
            return slicesData.map(x => new NdArray(x, sliceShape));
        }
        
            
        // Fill the slices with data.
        for (let i = 0; i < this.shape[dim]; i++) {
            slicesData[i] = new Array(sliceSize);
        }
        
        const chunkSize = this.shape.slice(dim+1).reduce((accumulator, currentValue) => accumulator * currentValue, 1);
        const totalChunkSize = numSlices * chunkSize;

        for (let i = 0; i < this.data.length; i++) {
            let sliceIdx = Math.floor((i % totalChunkSize) / chunkSize);
            let offset = Math.floor(i / totalChunkSize) * chunkSize + (i % chunkSize);
            // TODO: Determine if this would be faster with Array.push.
            slicesData[sliceIdx][offset] = this.data[i];
        }

        return slicesData.map(x => new NdArray(x, sliceShape));
    }

    public toUint8ClampedArray(dim: number | null = null): any {
        if (this.shape.length < 3 || this.shape.length > 4) {
            throw new Error(`Cannot convert ${this.shape.length}d array to an image.`);
        }

        if (this.shape.length === 4) {
            if (dim === null) {
                throw new Error('Cannot convert 3D image to images without a dimension specified.');
            }
            return this.slices(dim).map(slice => slice.toUint8ClampedArray());
        }

        // Base case: 2D array.
        if (this.shape.length === 3) {
            if (dim !== null) {
                throw new Error('Cannot convert a 2D array to an image with a dimension specified.');
            }
        }

        return new Uint8ClampedArray(this.data);



        // const imageData = new Uint8Array(this.data);
        // const string_char = imageData.reduce((data, byte) => {
        //     return data + String.fromCharCode(byte);
        // }, '');

        // return 'https://picsum.photos/200/300';
        //return `data:image/jpeg;base64,${btoa(string_char)}`;

    }

    public ndim(): number {
        return this.shape.length;
    }
}

export interface NdArrayInterface {
    data: Array<number>;
    shape: Array<number>;
}