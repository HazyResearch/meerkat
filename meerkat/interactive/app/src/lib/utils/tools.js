function isObject(elem) {
    return elem === Object(elem);
}

function objectMap(object, mapFn) {
    return Object.keys(object).reduce(function (result, key) {
        result[key] = mapFn(object[key])
        return result
    }, {})
}


export function nestedMap(elem, fn) {
    if (!elem) {
        return elem;
    }
    if (elem.is_store) {
        return fn(elem)
    }

    if (Array.isArray(elem)) {
        return elem.map((x) => nestedMap(x, fn))
    } else if (isObject(elem)) {
        return objectMap(elem, (x) => nestedMap(x, fn))
    } else {
        return fn(elem)
    }
}

