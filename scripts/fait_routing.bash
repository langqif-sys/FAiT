# Shared options for the FAiT dataset launchers (run from the project root).
configure_fait_routing() {
    local default_log_dir=$1
    shift
    if (( $# > 2 )); then
        echo "Usage: bash $0 [off|signed|abs|energy] [96|192|336|720]" >&2
        return 2
    fi
    route_mode=${1:-${FAIT_SPECTRAL_ROUTING:-off}}
    only_horizon=${2:-all}
    case "$route_mode" in
        off|signed|abs|energy) ;;
        *) echo "Unknown spectral routing mode: $route_mode" >&2; return 2 ;;
    esac
    case "$only_horizon" in
        all|96|192|336|720) ;;
        *) echo "Unsupported forecast horizon: $only_horizon" >&2; return 2 ;;
    esac
    log_dir="$default_log_dir"
    if [[ "$route_mode" != off ]]; then
        log_dir="./logs/FAiT/spectral_routing/$route_mode"
    fi
    mkdir -p "$log_dir"
    printf 'FAiT spectral_routing=%s  horizon=%s\n' "$route_mode" "$only_horizon"
}
