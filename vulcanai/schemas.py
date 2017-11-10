"""The following schemas were created from https://jsonschema.net/#/editor."""

dense_network = {
    "properties": {
        "dropouts": {
            "id": "/properties/dropouts",
            "items": {
                "id": "/properties/dropouts/items",
                "type": "number",
                "maximum": 1,
                "minimum": 0
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
                "type": "integer",
                "minimum": 0
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
                    "type": "integer",
                    "minimum": 1
                },
                "type": "array"
            },
            "type": "array"
        },
        "filters": {
            "id": "/properties/filters",
            "items": {
                "id": "/properties/filters/items",
                "type": "integer",
                "minimum": 1
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
                    "type": "string",
                    "enum": ["max", "average_inc_pad", "average_exc_pad"]
                },
                "stride": {
                    "id": "/properties/pool/properties/stride",
                    "items": {
                        "id": "/properties/pool/properties/stride/items",
                        "items": {
                            "id": "/properties/pool/properties/stride/items/items",
                            "type": "integer",
                            "minimum": 1
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
                    "type": "integer",
                    "minimum": 1
                },
                "type": "array"
            },
            "type": "array"
        }
    },
    "type": "object"
}
