import config
import logging
import nst
import os

from aiogram import Bot, Dispatcher, executor, types
from aiogram.utils.executor import start_webhook

from shutil import copyfile


# webhook settings
# WEBHOOK_HOST = 'https://your.domain'
# WEBHOOK_PATH = '/path/to/api'
# WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

# webserver settings
# WEBAPP_HOST = 'localhost'  # or ip
# WEBAPP_PORT = 3001

# port
# PORT = int(os.environ.get('PORT', 5000))

# logging level
logging.basicConfig(level=logging.INFO)

# initialize bot
bot = Bot(token=config.TOKEN)
dp = Dispatcher(bot)


# start message
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    sti = open('images/source/fallout.webp', 'rb')
    await bot.send_sticker(message.chat.id, sti)

    # keyboard
    markup_general = types.ReplyKeyboardMarkup(resize_keyboard=True)
    button3_general = types.KeyboardButton('Прикрепить фото 🚀')
    button4_general = types.KeyboardButton('Покажи пример 🤷️')

    markup_general.add(button3_general, button4_general)

    await bot.send_message(message.chat.id, 'Здравствуй, {0.first_name}!\nЯ - <b>{1.first_name}</b>,  '
                                            'робот, который магически раскрасит твою картинку 🎨'
                                            ''.format(message.from_user, await bot.get_me()),
                           parse_mode='html', reply_markup=markup_general)
    await bot.send_message(message.chat.id, 'Есть вопросы? ➡️ /help')


# help
@dp.message_handler(commands=['help'])
async def help_message(message: types.Message):
    await bot.send_message(message.chat.id, 'Я робот, который может перенести стиль с одной фотографии на другую.'
                                            'Присылай мне фото, которое хочешь преобразовать.'
                                            'лишь нужно отправить мне фото, которое ты хочешь преобразить. Далее, '
                                            'тебе будет предложено несколько стилей на выбор. Чтобы посмотреть пример '
                                            'обработки, можешь ткнуть на кнопку "Покажи пример"')


# chat
@dp.message_handler(content_types=['text'])
async def chat(message: types.Message):
    if message.chat.type == 'private':
        if message.text == 'Прикрепить фото 🚀':
            await bot.send_message(message.chat.id,
                                   'Пожалуйста, отправьте мне фото, которое хотите обработать. '
                                   'Лучше отправьте фото так, чтобы объект на фото располагался по центру, так как в процессе '
                                   'магии фото кадрируется.')
        elif message.text == 'Покажи пример 🤷️':
            content_pic = open('images/source/dog.jpg', 'rb')
            result_pic = open('images/results/result24.jpg', 'rb')

            reply_markup = types.InlineKeyboardMarkup(row_width=2, one_time_keyboard=True)


            await bot.send_photo(message.chat.id, content_pic)
            await bot.send_message(message.chat.id, 'Например, ты отправляешь фото собачки 🐶️')
            await bot.send_photo(message.chat.id, result_pic)
            await bot.send_message(message.chat.id, '...а я делаю из нее цветную собаку! 🤡 ️',
                                   reply_markup=reply_markup)

        else:
            await bot.send_message(message.chat.id, 'Моя твоя не понимать 😢')


# save photo
@dp.message_handler(content_types=['photo'])
async def handle_photo(message: types.Message):
    await message.photo[-1].download('./images/source/content_image.jpg')
    await bot.send_message(message.chat.id, 'Фото успешно загружено!')

    style_reply_markup = types.InlineKeyboardMarkup(row_width=2, one_time_keyboard=True)
    style_reply_button1 = types.InlineKeyboardButton('Комиксы', callback_data='marvel')
    style_reply_button2 = types.InlineKeyboardButton('Пикассо', callback_data='picasso')
    style_reply_button3 = types.InlineKeyboardButton('Лицо', callback_data='face')
    style_reply_button4 = types.InlineKeyboardButton('Лего', callback_data='lego')
    style_reply_button5 = types.InlineKeyboardButton('Конфетки', callback_data='sweet')
    style_reply_button6 = types.InlineKeyboardButton('Макароны', callback_data='makaroni')

    style_reply_markup.add(style_reply_button1,
                           style_reply_button2,
                           style_reply_button3,
                           style_reply_button4,
                           style_reply_button5,
                           style_reply_button6)

    media = types.MediaGroup()
    media.attach_photo(types.InputFile('images/source/marvel.jpg'), 'Это Комиксы Marvel')
    media.attach_photo(types.InputFile('images/source/picasso.jpg'), 'Это Пикасco')
    media.attach_photo(types.InputFile('images/source/face.jpg'), 'Это Лицо')
    media.attach_photo(types.InputFile('images/source/lego.jpg'), 'Это Лего')
    media.attach_photo(types.InputFile('images/source/sweet.jpg'), 'Это Конфетки')
    media.attach_photo(types.InputFile('images/source/makaroni.jpg'), 'Это Макароны')


    await bot.send_media_group(chat_id=message.chat.id, media = media)
    await bot.send_message(message.chat.id, 'Выбирайте стиль! 🖼:', reply_markup=style_reply_markup)


# callback
@dp.callback_query_handler(lambda call: True)
async def callback_inline(call):
    try:
        if call.message:
            # style
            if call.data == 'marvel':
                await bot.send_message(call.message.chat.id, 'Прекрасно!')
                copyfile('images/source/marvel.jpg', 'images/source/style_image.jpg')
                await launch_nst(call.message)
            elif call.data == 'picasso':
                await bot.send_message(call.message.chat.id, 'Замечательно!')
                copyfile('images/source/picasso.jpg', 'images/source/style_image.jpg')
                await launch_nst(call.message)
            elif call.data == 'face':
                await bot.send_message(call.message.chat.id, 'Хороший выбор!')
                copyfile('images/source/face.jpg', 'images/source/style_image.jpg')
                await launch_nst(call.message)
            elif call.data == 'lego':
                await bot.send_message(call.message.chat.id, 'Топ!')
                copyfile('images/source/lego.jpg', 'images/source/style_image.jpg')
                await launch_nst(call.message)
            elif call.data == 'sweet':
                await bot.send_message(call.message.chat.id, 'Прекрасно!')
                copyfile('images/source/sweet.jpg', 'images/source/style_image.jpg')
                await launch_nst(call.message)
            elif call.data == 'makaroni':
                await bot.send_message(call.message.chat.id, 'Хороший выбор!')
                copyfile('images/source/makaroni.jpg', 'images/source/style_image.jpg')
                await launch_nst(call.message)

            # remove inline buttons
            await bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, text='...',
                                        reply_markup=None)
    except Exception as e:
        print(repr(e))


# launch style transfer
async def launch_nst(message):
    # print('ok')
    content_image_name = 'images/source/content_image.jpg'
    style_image_name = 'images/source/style_image.jpg'

    await bot.send_message(message.chat.id, 'Начинаем обработку фотографии. Это займёт совсем немного времени!) 🔮️')

    nst.main(content_image_name, style_image_name)

    await bot.send_message(message.chat.id, 'Готово!')

    result = open('images/results/bot-result.png', 'rb')

    await bot.send_photo(message.chat.id, result)

    await bot.send_message(message.chat.id, 'Чтобы попробовать ещё раз, просто отправь мне новое фото. 💫')


# async def on_startup(dp):
#    await bot.set_webhook(WEBHOOK_URL)
    # insert code here to run it after start


# async def on_shutdown(dp):
#    logging.warning('Shutting down..')

    # insert code here to run it before shutdown

    # Remove webhook (not acceptable in some cases)
#    await bot.delete_webhook()

    # Close DB connection (if used)
#    await dp.storage.close()
#    await dp.storage.wait_closed()

 #   logging.warning('Bye!')


# launch long polling
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
    #start_webhook(listen='0.0.0.0', port=int(PORT), url_path=config.TOKEN)
    #executor.set_webhook('https://style-tg-bot.herokuapp.com/' + config.TOKEN)
    #start_webhook(
    #    dispatcher=dp,
    #    webhook_path=WEBHOOK_PATH,
    #    on_startup=on_startup,
    #    on_shutdown=on_shutdown,
    #    skip_updates=True,
    #    host=WEBAPP_HOST,
    #    port=WEBAPP_PORT,
    #)
