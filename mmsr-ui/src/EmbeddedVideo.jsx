import { Paper } from '@mui/material';

export default function EmbeddedVideo(props) {
    const url = props.url;
    const embedUrl = transformToEmbedLink(url);

    function transformToEmbedLink(url) {
        const videoId = url.split('v=')[1];
        if (videoId) {
            const embedLink = `https://www.youtube.com/embed/${videoId}`;
            return embedLink;
        } else {
            return 'Invalid YouTube link';
        }
    }

    transformToEmbedLink(url);
    return (
        <Paper elevation={3} style={{ width: '100%', margin: 'auto' }}>
            <iframe
                title="YouTube Video"
                src={embedUrl}
                allowFullScreen
            />
        </Paper>
    );
};
