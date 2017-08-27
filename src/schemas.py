"""The following schemas were created from https://jsonschema.net/#/editor."""

dense_network = {
    "properties": {
        "dropouts": {
            "id": "/properties/dropouts",
            "items": {
                "id": "/properties/dropouts/items",
                "type": "number"
            },
            "type": "array"
        },
        "mode": {
            "id": "/properties/mode",
            "type": "string"
        },
        "units": {
            "id": "/properties/units",
            "items": {
                "id": "/properties/units/items",
                "type": "integer"
            },
            "type": "array"
        }
    },
    "type": "object"
}

conv_network = {
    "properties": {
        "filter_size": {
            "id": "/properties/filter_size",
            "items": {
                "id": "/properties/filter_size/items",
                "items": {
                    "id": "/properties/filter_size/items/items",
                    "type": "integer"
                },
                "type": "array"
            },
            "type": "array"
        },
        "filters": {
            "id": "/properties/filters",
            "items": {
                "id": "/properties/filters/items",
                "type": "integer"
            },
            "type": "array"
        },
        "mode": {
            "id": "/properties/mode",
            "type": "string"
        },
        "pool": {
            "id": "/properties/pool",
            "properties": {
                "mode": {
                    "id": "/properties/pool/properties/mode",
                    "type": "string"
                },
                "stride": {
                    "id": "/properties/pool/properties/stride",
                    "items": {
                        "id": "/properties/pool/properties/stride/items",
                        "items": {
                            "id": "/properties/pool/properties/stride/items/items",
                            "type": "integer"
                        },
                        "type": "array"
                    },
                    "type": "array"
                }
            },
            "type": "object"
        },
        "stride": {
            "id": "/properties/stride",
            "items": {
                "id": "/properties/stride/items",
                "items": {
                    "id": "/properties/stride/items/items",
                    "type": "integer"
                },
                "type": "array"
            },
            "type": "array"
        }
    },
    "type": "object"
}
