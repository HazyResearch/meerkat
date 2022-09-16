

function isObject(elem: any) {
    return elem === Object(elem);
}

function objectMap(object: Object, mapFn: Function) {
    return Object.keys(object).reduce(function (result: Object, key: string) {
        result[key] = mapFn(object[key])
        return result
    }, {})
}


export function nestedMap(elem: any, fn: Function): any {
    if (!elem) {
        return elem; 
    }
    if (elem.is_store) {
        return fn(elem)
    }

    if (Array.isArray(elem)) {
        return elem.map((x: any) => nestedMap(x, fn))
    } else if (isObject(elem)) {
        return objectMap(elem, (x: any) => nestedMap(x, fn))
    } else {
        return fn(elem)
    }
}

