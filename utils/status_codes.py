class StatusCodes:
    # Task stage
    SUCCESS         = "PII_000"
    PENDING         = "PII_001" 
    INPROGRESS      = "PII_002"
    
    # Client
    INVALID_REQUEST                 =  "PII_400" # Empty, null, invalid filed in payload
    EXCEEDING_PERMITTED_RESOURCES   = "PII_401"  # < 300s is permitted
    RESOURCE_DOES_NOT_EXIST         = "PII_402"  # Can not find melody for exmaple
    UNSUPPORTED                     = "PII_403"  # Type of resource: melody must be *mp3

    # Server
    TIMEOUT         = "PII_500" # If a task exceeding timeout => Set status timeout
    ERROR           = "PII_501" # unknown ERROR
    RABBIT_ERROR    = "PII_502" # Service cannot connect to Rabbit
    REDIS_ERROR     = "PII_303" # Service cannot connect to Redis
    S3_ERROR        = "PII_504" # Service cannot connect to S3