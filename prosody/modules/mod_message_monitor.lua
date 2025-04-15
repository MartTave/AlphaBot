-- A Prosody module that captures all messages and forwards them to an HTTP endpoint

local http = require "net.http";
local json = require "util.json";
local uuid = require "util.uuid";

-- Configuration
local api_endpoint = module:get_option_string("message_monitor_endpoint", "http://172.20.0.3:3000/api/messages");
local api_token = module:get_option_string("message_monitor_token", "your_secret_token");

module:log("info", "Message monitor initialized. Endpoint: %s", api_endpoint);

-- Helper function to process and send message data
local function process_message(stanza, direction)
    local from = stanza.attr.from;
    local to = stanza.attr.to;
    local body = stanza:get_child_text("body");
    local message_type = stanza.attr.type or "normal";
    local message_id = stanza.attr.id or uuid.generate();

    -- Only process messages with body content
    if body then
        local message_data = {
            id = message_id,
            from = from,
            to = to,
            body = body,
            type = message_type,
            direction = direction,
            timestamp = os.time(),
            conversation_id = from < to and (from .. "-" .. to) or (to .. "-" .. from)
        }

        -- Send data to API endpoint
        http.request(api_endpoint, {
            method = "POST",
            body = json.encode(message_data),
            headers = {
                ["Content-Type"] = "application/json",
                ["Authorization"] = "Bearer " .. api_token
            }
        }, function(response_body, response_code)
            if response_code ~= 200 then
                module:log("error", "Failed to send message to monitor API: %d", response_code);
            end
        end);

        module:log("debug", "Forwarded %s message from %s to %s", direction, from, to);
    end
end

-- Hook for incoming messages
module:hook("message/full", function(event)
    process_message(event.stanza, "received");
    return nil;
end, 10);

-- Hook for outgoing messages
module:hook("message/bare", function(event)
    process_message(event.stanza, "sent");
    return nil;
end, 10);

module:log("info", "Message monitor module loaded successfully");
